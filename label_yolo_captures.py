"""Interactive multi-class circle/line annotation editor."""
from __future__ import annotations

from shellshock.config.paths import DATA_ROOT

import argparse, json, shutil
from dataclasses import dataclass
from datetime import datetime
from math import cos, hypot, radians, sin
from pathlib import Path
from time import monotonic
import cv2
import keyboard

from shellshock.datasets.export import yolo_label_line
from shellshock.datasets.yolo import CLASS_NAMES, YoloBox, parse_yolo_label_text
from shellshock.perception.color_geometry import detect_pink_obstacle_geometry
from shellshock.perception.yolo import YoloDetector
from shellshock.annotations.conversion import scene_from_editor, save_manual_scene
from shellshock.application.scene import analyze_frame
from shellshock.application.solver import solve_integer_shot
from shellshock.planning.policies import select_mode
from shellshock.rendering.trajectory import sample_solution_trajectory
from shellshock.rendering.overlay import draw_trajectory
from replay_shellshock import (
    arrow_adjustment,
    guide_from_report,
    manual_controls,
    render_replay_visual,
    replay_image,
    wind_adjustment,
)

ANNOTATION_MODE = "annotation"
REPLAY_MODE = "replay"
REPLAY_SOLVER_MODES = {
    ord("1"): "normal_low",
    ord("2"): "reflection_low",
    ord("3"): "wormhole_low",
}


def toggle_interaction_mode(mode):
    return REPLAY_MODE if mode == ANNOTATION_MODE else ANNOTATION_MODE


def cycle_image_index(index, delta, image_count):
    if image_count <= 0:
        raise ValueError("image_count must be positive")
    return (int(index) + int(delta)) % int(image_count)


def interaction_key_action(mode, key):
    """Describe mode-specific keys without touching the GUI or annotation state."""
    key = int(key) & 0xff
    normalized_key = key + (ord("a") - ord("A")) if ord("A") <= key <= ord("Z") else key
    if mode == ANNOTATION_MODE:
        if ord("0") <= key <= ord("9"):
            return ("select_class", key - ord("0"))
        if key in (ord("+"), ord("=")):
            return ("resize", 2)
        if key in (ord("-"), ord("_")):
            return ("resize", -2)
        return None
    if key in REPLAY_SOLVER_MODES:
        return ("solver_mode", REPLAY_SOLVER_MODES[key])
    if normalized_key == ord("v"):
        return ("toggle_annotations",)
    if key == 9:
        return ("toggle_tuning",)
    return None


def format_replay_log_entry(report, trigger="ALT"):
    """Format one in-memory replay calculation with all solver diagnostics."""
    solution = report.get("solution", {}) if isinstance(report, dict) else {}
    lines = [
        f"timestamp={solution.get('_log_timestamp', datetime.now().isoformat(timespec='seconds'))} trigger={trigger}",
        f"image={report.get('image', '')} mode={solution.get('mode', '')} target={report.get('target')}",
        f"status={solution.get('status', '')} direction={solution.get('direction', '')} angle={solution.get('angle_degrees', '')} power={solution.get('power', '')}",
    ]
    lines.extend(report.get("debug_logs", ()) if isinstance(report, dict) else ())
    trace = solution.get("final_trace", {})
    if trace:
        lines.append(f"SELECTED route_id={solution.get('selected_route_id')} branch_id={solution.get('selected_branch_id')} candidate_id={solution.get('selected_candidate_id')}")
        for waypoint in trace.get("waypoints", ()):
            lines.append(f"WAYPOINT kind={waypoint.get('kind')} id={waypoint.get('id', waypoint.get('object_id', ''))} time={waypoint.get('time')} point={waypoint.get('point')}")
        for segment in trace.get("segments", ()):
            lines.append(f"SEGMENT {segment.get('index')} start={segment.get('start')} end={segment.get('end')} velocity={segment.get('velocity')} acceleration={segment.get('acceleration')} duration={segment.get('duration')} elapsed={segment.get('elapsed')}")
        for transition in trace.get("portal_transitions", ()):
            lines.append(f"PORTAL_TRANSITION id={transition.get('portal_id')} entry_point={transition.get('entry_point')} exit_point={transition.get('exit_point')} displacement={transition.get('displacement')}")
    return "\n".join(lines)


def replay_log_status(entries):
    return "LOG SAVED" if entries else "NO LOGS TO SAVE"


def save_replay_log_bundle(destination, entries):
    destination = Path(destination)
    if not entries:
        return "NO LOGS TO SAVE"
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = [item if "report" in item else {"report": item, "trigger": item.get("trigger", "ALT"), "timestamp": item.get("timestamp")} for item in entries]
    readable = []
    for item in normalized:
        report = dict(item["report"])
        if item.get("timestamp") is not None:
            report["solution"] = {**report.get("solution", {}), "_log_timestamp": item["timestamp"]}
        readable.append(format_replay_log_entry(report, item.get("trigger", "ALT")))
    destination.write_text("\n\n".join(readable) + "\n", encoding="utf-8")
    destination.with_suffix(".json").write_text(json.dumps(normalized, ensure_ascii=False, indent=2, default=list) + "\n", encoding="utf-8")
    return f"LOG SAVED: {destination.name} + .json"


def annotation_wind_metadata(metadata, wind_signed):
    """Return image annotation metadata for a signed, manually edited wind."""
    payload = dict(metadata or {})
    signed = float(wind_signed)
    payload["wind_value"] = round(abs(signed), 2)
    payload["wind_direction"] = "left" if signed < 0 else "right"
    payload["wind_source"] = "manual"
    return payload


def format_wind_text(wind_signed):
    signed = float(wind_signed)
    return f"{abs(signed):.0f} {'LEFT' if signed < 0 else 'RIGHT'}"

CLASS_KEY_MAP = {str(i): i for i in range(10)}
CLASS_LABELS = {0:"enemy / 敌人",1:"self_center_keypoint / 新己方中心关键点",2:"self / 己方中心",3:"obstacle_circle / 圆形障碍物",4:"obstacle_line / 线段障碍物",5:"portal_orange / 橙色虫洞",6:"portal_blue / 蓝色虫洞",7:"blackhole / 黑洞",8:"double_damage / 二倍伤害",9:"Triple_damage / 三倍伤害"}
SELF_DISPLAY_SCALE = 1.75

@dataclass(frozen=True)
class CircleAnnotation:
    class_id: int; center_x: float; center_y: float; radius: float

@dataclass(frozen=True)
class LineAnnotation:
    class_id: int; start: tuple[float,float]; end: tuple[float,float]

def circle_to_yolo_box(a: CircleAnnotation, image_width: int, image_height: int) -> str:
    left, top = max(0.,a.center_x-a.radius), max(0.,a.center_y-a.radius)
    right, bottom = min(float(image_width),a.center_x+a.radius), min(float(image_height),a.center_y+a.radius)
    if right<=left or bottom<=top: raise ValueError("annotation is outside the image")
    return f"{a.class_id} {(left+right)/2/image_width:.6f} {(top+bottom)/2/image_height:.6f} {(right-left)/image_width:.6f} {(bottom-top)/image_height:.6f}"

def _circle_intersects_image(a: CircleAnnotation, image_width: int, image_height: int) -> bool:
    return (
        max(0.0, a.center_x - a.radius) < image_width
        and min(float(image_width), a.center_x + a.radius) > 0.0
        and max(0.0, a.center_y - a.radius) < image_height
        and min(float(image_height), a.center_y + a.radius) > 0.0
    )

def point_to_yolo_box(class_id, point, image_width, image_height, radius=8.0) -> str:
    """Store a keypoint in the regular YOLO label file as a small box."""
    return circle_to_yolo_box(
        CircleAnnotation(class_id, float(point[0]), float(point[1]), float(radius)),
        image_width,
        image_height,
    )

def line_to_yolo_box(a: LineAnnotation, image_width: int, image_height: int, padding: float=3.) -> str:
    x0,y0=a.start; x1,y1=a.end; left=max(0.,min(x0,x1)-padding); top=max(0.,min(y0,y1)-padding); right=min(float(image_width),max(x0,x1)+padding); bottom=min(float(image_height),max(y0,y1)+padding)
    return f"{a.class_id} {(left+right)/2/image_width:.6f} {(top+bottom)/2/image_height:.6f} {(right-left)/image_width:.6f} {(bottom-top)/image_height:.6f}"

def delete_nearest_annotation(annotations, x, y, class_id):
    candidates=[(i,a) for i,a in enumerate(annotations) if isinstance(a,CircleAnnotation) and a.class_id==class_id and hypot(a.center_x-x,a.center_y-y)<=a.radius]
    if not candidates: return list(annotations)
    i,_=min(candidates,key=lambda item:hypot(item[1].center_x-x,item[1].center_y-y)); return annotations[:i]+annotations[i+1:]

def _point_segment_distance(x, y, start, end):
    sx, sy = start; ex, ey = end
    dx, dy = ex - sx, ey - sy
    if dx == 0 and dy == 0: return hypot(x - sx, y - sy)
    t = max(0., min(1., ((x - sx) * dx + (y - sy) * dy) / (dx * dx + dy * dy)))
    return hypot(x - (sx + t * dx), y - (sy + t * dy))

def delete_nearest_line(lines, x, y, max_distance=14.):
    candidates=[(i,_point_segment_distance(x,y,a.start,a.end)) for i,a in enumerate(lines)]
    candidates=[item for item in candidates if item[1]<=max_distance]
    if not candidates: return list(lines)
    i,_=min(candidates,key=lambda item:item[1]); return lines[:i]+lines[i+1:]

def _merge_line_annotations(lines, endpoint_tolerance=24.0, angle_tolerance=0.12):
    """Collapse near-identical line geometry while retaining the first/manual line."""
    merged = []
    for candidate in lines:
        cx, cy = candidate.start
        ex, ey = candidate.end
        length = max(1.0, hypot(ex - cx, ey - cy))
        matched = False
        for existing in merged:
            sx, sy = existing.start
            tx, ty = existing.end
            other_length = max(1.0, hypot(tx - sx, ty - sy))
            cross = abs((ex - cx) * (ty - sy) - (ey - cy) * (tx - sx)) / (length * other_length)
            direct = hypot(cx - sx, cy - sy) + hypot(ex - tx, ey - ty)
            reverse = hypot(cx - tx, cy - ty) + hypot(ex - sx, ey - sy)
            if cross <= angle_tolerance and min(direct, reverse) <= max(endpoint_tolerance, 0.15 * max(length, other_length)):
                matched = True
                break
        if not matched:
            merged.append(candidate)
    return merged


def _merge_duplicate_line_boxes(boxes, overlap_threshold=0.60):
    """Keep one YOLO line box when multiple predictions cover the same board."""
    merged = []
    for candidate in boxes:
        if candidate.class_id != 4:
            merged.append(candidate)
            continue
        candidate_left = candidate.center_x - candidate.width / 2
        candidate_top = candidate.center_y - candidate.height / 2
        candidate_right = candidate.center_x + candidate.width / 2
        candidate_bottom = candidate.center_y + candidate.height / 2
        match = None
        for index, existing in enumerate(merged):
            if existing.class_id != 4:
                continue
            existing_left = existing.center_x - existing.width / 2
            existing_top = existing.center_y - existing.height / 2
            existing_right = existing.center_x + existing.width / 2
            existing_bottom = existing.center_y + existing.height / 2
            intersection = max(0., min(candidate_right, existing_right) - max(candidate_left, existing_left)) * max(0., min(candidate_bottom, existing_bottom) - max(candidate_top, existing_top))
            smaller_area = min(candidate.width * candidate.height, existing.width * existing.height)
            if smaller_area > 0 and intersection / smaller_area >= overlap_threshold:
                match = index
                break
        if match is None:
            merged.append(candidate)
            continue
        existing = merged[match]
        left, top = min(candidate_left, existing.center_x - existing.width / 2), min(candidate_top, existing.center_y - existing.height / 2)
        right, bottom = max(candidate_right, existing.center_x + existing.width / 2), max(candidate_bottom, existing.center_y + existing.height / 2)
        merged[match] = YoloBox(4, (left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
    return merged

def select_images(raw_dir: Path, start: str, end: str):
    return sorted((p for p in Path(raw_dir).iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.webp'} and start<=p.stem<=end),key=lambda p:p.name)

def display_to_image_point(point, display_size, image_size): return round(point[0]*image_size[0]/display_size[0]),round(point[1]*image_size[1]/display_size[1])
def key_action(key):
    if key in (10, 13): return 'accept'
    if key in (81,2424832): return 'previous'
    if key in (83,2555904): return 'next'
    if key == 2162688: return 'previous'
    if key == 2228224: return 'next'
    if key==27: return 'quit'

def _display_size(w,h):
    scale=min(1.,1600/w,1000/h); return max(1,round(w*scale)),max(1,round(h*scale))

def dynamic_barrel_length(image_width, base_length):
    """Scale the 2560-pixel reference barrel length to the current image."""
    return float(base_length) * float(image_width) / 2560.0

def _circle_at(circles, x, y, preferred_class=None):
    candidates=[]
    for i, a in enumerate(circles):
        if preferred_class is not None and a.class_id != preferred_class: continue
        distance=hypot(a.center_x-x,a.center_y-y)
        hit_radius=a.radius*SELF_DISPLAY_SCALE if a.class_id==2 else a.radius
        if distance <= max(hit_radius,8.): candidates.append((distance,i))
    return min(candidates)[1] if candidates else None

def _center_click_action(selected, center, circles, x, y):
    """Return the action for a left click while the center-keypoint tool is active."""
    if selected != 1:
        return None
    if center is not None and hypot(center[0] - x, center[1] - y) <= 16.:
        return "select_center"
    return "set_center"

def _adjust_circle_radius(circles, selected_object, delta):
    """Adjust a selected circle, clearing stale selections safely."""
    if not selected_object or selected_object[0] != 'circle':
        return circles, selected_object
    index=selected_object[1]
    if not isinstance(index,int) or not 0 <= index < len(circles):
        return circles, None
    updated=list(circles)
    a=updated[index]
    updated[index]=CircleAnnotation(a.class_id,a.center_x,a.center_y,max(4,min(1000,a.radius+delta)))
    return updated, selected_object

def _wheel_delta(flags):
    """Read Windows/OpenCV wheel direction across different backend builds."""
    getter=getattr(cv2,'getMouseWheelDelta',None)
    if getter is not None:
        try:
            delta=int(getter(flags))
            if delta: return delta
        except (TypeError,ValueError):
            pass
    raw=(int(flags)>>16) & 0xffff
    return raw-0x10000 if raw & 0x8000 else raw
def _load_boxes(path,w,h):
    if not path.exists(): return []
    boxes,errors=parse_yolo_label_text(path.read_text(encoding='utf-8'),w,h)
    if errors: raise ValueError(f'invalid labels in {path}: {" | ".join(errors)}')
    return boxes
def _metadata(path):
    if not path.exists(): return None
    return json.loads(path.read_text(encoding='utf-8'))

def _save_metadata_angle(path, metadata, angle):
    if angle is None: return
    payload=dict(metadata or {})
    payload['angle_degrees']=round(float(angle),2)
    direction=payload.get('direction','right')
    sign=-1. if direction=='left' else 1.
    angle_rad=radians(float(angle))
    payload['direction_vector']=[sign*cos(angle_rad),-sin(angle_rad)]
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def _save_wind_label(label_dir, wind_crops_dir, stem, wind_signed, source='detected'):
    crop_path=Path(wind_crops_dir)/f'{stem}.png'
    signed=round(float(wind_signed),2)
    payload={'image':str(crop_path), 'stem':stem, 'wind_value':abs(signed), 'wind_direction':'left' if signed<0 else 'right', 'wind_signed':signed, 'source':source}
    path=Path(label_dir)/f'{stem}.json'; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def _line_from_box(box,w,h):
    cx,cy=box.center_x*w,box.center_y*h; width,height=box.width*w,box.height*h
    if width>=height: return LineAnnotation(4,(cx-width/2,cy),(cx+width/2,cy))
    return LineAnnotation(4,(cx,cy-height/2),(cx,cy+height/2))


def _select_color_geometry(color_geometry, pose_geometry, box_geometry):
    """Prefer white-pixel geometry, then pose geometry, then the YOLO box."""
    if color_geometry:
        return color_geometry[0]
    if pose_geometry:
        return pose_geometry[0]
    return box_geometry

def _save_state(override_path, geometry_path, circles, lines, width, height, center, muzzle):
    scene=scene_from_editor(circles,lines,width,height,center,muzzle)
    save_manual_scene(scene,override_path,geometry_path,backup=False)


def _move_if_exists(source: Path, destination: Path):
    if not source.exists(): return
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists(): raise FileExistsError(f'annotation destination already exists: {destination}')
    shutil.move(str(source),str(destination))

def move_annotation_bundle(image_path, override_path, geometry_path, metadata_path, preview_path, wind_label_path, annotation_dir):
    """Move one completed annotation and its sidecars into the reviewed tree."""
    root=Path(annotation_dir)
    stem=Path(image_path).stem
    pairs=((Path(image_path),root/'images'/Path(image_path).name),
           (Path(override_path),root/'labels'/f'{stem}.txt'),
           (Path(geometry_path),root/'pose_geometry'/f'{stem}.json'),
           (Path(metadata_path),root/'metadata'/f'{stem}.json'),
           (Path(preview_path),root/'previews'/f'{stem}.jpg'),
           (Path(wind_label_path),root/'wind_labels'/f'{stem}.json'))
    for source,destination in pairs:
        if source.exists() and destination.exists():
            raise FileExistsError(f'annotation destination already exists: {destination}')
    for source,destination in pairs:
        _move_if_exists(source,destination)

def _draw(image,circles,lines,center,muzzle,square=False,pose_detections=()):
    out=image.copy(); colors=[(0,0,255),(0,200,0),(255,0,0),(0,200,255),(255,255,0),(0,140,255),(255,120,0),(180,0,255),(0,255,180),(255,0,180)]
    for detection in pose_detections:
        for index, point in enumerate(getattr(detection, 'keypoints', ())):
            if detection.name == 'self' and index == 0: continue
            if point.visible <= 0: continue
            p=(round(point.x),round(point.y)); color=(255,255,255) if index == 0 else (255,120,255)
            cv2.circle(out,p,7,color,-1); cv2.circle(out,p,9,(0,0,0),1)
            cv2.putText(out,f'{detection.name}.kp{index}',(p[0]+8,p[1]-8),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1)
    for a in circles:
        c=colors[a.class_id%len(colors)]; p=(round(a.center_x),round(a.center_y))
        display_radius=a.radius*SELF_DISPLAY_SCALE if a.class_id==2 else a.radius
        if square: cv2.rectangle(out,(round(a.center_x-display_radius),round(a.center_y-display_radius)),(round(a.center_x+display_radius),round(a.center_y+display_radius)),c,3)
        else: cv2.circle(out,p,max(1,round(display_radius)),c,3)
        cv2.putText(out,CLASS_LABELS.get(a.class_id,str(a.class_id)),(p[0]-45,max(18,p[1]-round(display_radius)-6)),cv2.FONT_HERSHEY_SIMPLEX,.48,c,2)
    for a in lines:
        c=colors[4]; s=(round(a.start[0]),round(a.start[1])); e=(round(a.end[0]),round(a.end[1])); cv2.line(out,s,e,c,3); cv2.circle(out,s,6,c,-1); cv2.circle(out,e,6,c,-1); cv2.putText(out,'obstacle_line',(s[0],max(18,s[1]-8)),cv2.FONT_HERSHEY_SIMPLEX,.48,c,2)
    if muzzle: cv2.circle(out,(round(muzzle[0]),round(muzzle[1])),8,(255,0,255),-1); cv2.putText(out,'muzzle',(round(muzzle[0])+8,round(muzzle[1])),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,255),2)
    if center:
        x,y=round(center[0]),round(center[1]); cv2.drawMarker(out,(x,y),(255,0,255),cv2.MARKER_CROSS,22,3); cv2.putText(out,'self_center',(x+10,y-10),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,255),2)
    return out

def run_annotation(images,raw_dir,override_dir,preview_dir,geometry_dir,metadata_dir,barrel_length,wind_label_dir,wind_crops_dir,annotation_dir,weights=None,confidence=.35):
    if not images: raise ValueError('no images matched the selected timestamp range')
    index,selected,radius,square=0,0,24.,False
    pose_detector=YoloDetector(str(weights), confidence) if weights else None
    while 0<=index<len(images):
        path=images[index]; image=cv2.imread(str(path)); h,w=image.shape[:2]; override=Path(override_dir)/f'{path.stem}.txt'; geometry_path=Path(geometry_dir)/f'{path.stem}.json'; metadata_path=Path(metadata_dir)/f'{path.stem}.json'; meta=_metadata(metadata_path); angle=float(meta['angle_degrees']) if meta and meta.get('angle_degrees') is not None else None
        wind_value=float((meta or {}).get('wind_value', 0))
        wind_direction=str((meta or {}).get('wind_direction', 'right')).lower()
        wind_signed=abs(wind_value) * (-1. if wind_direction=='left' else 1.)
        meta=dict(meta or {})
        meta['wind_value']=round(abs(wind_signed),2)
        meta['wind_direction']='left' if wind_signed<0 else 'right'
        meta['wind_source']=str(meta.get('wind_source','manual'))
        boxes=_load_boxes(override if override.exists() else Path(raw_dir)/f'{path.stem}.txt',w,h); circles=[]; lines=[]
        pose_detections=pose_detector.detect(image) if pose_detector else []
        def pose_for(class_name, box):
            candidates=[item for item in pose_detections if item.name == class_name]
            if not candidates: return None
            cx,cy=box.center_x*w,box.center_y*h
            return min(candidates,key=lambda item:hypot(item.x+item.width/2-cx,item.y+item.height/2-cy))
        saved_geometry={}
        if geometry_path.exists():
            try:
                saved_geometry=json.loads(geometry_path.read_text(encoding='utf-8'))
            except (OSError, ValueError, TypeError):
                saved_geometry={}
        saved_circles=[
            CircleAnnotation(int(item['class_id']),float(item['center'][0]),float(item['center'][1]),float(item['radius']))
            for item in saved_geometry.get('circles',[])
            if isinstance(item,dict) and 'class_id' in item and 'center' in item and len(item['center']) == 2 and 'radius' in item
        ]
        circles=list(saved_circles)
        saved_lines=[LineAnnotation(4,tuple(item['start']),tuple(item['end'])) for item in saved_geometry.get('lines',[]) if isinstance(item,dict) and 'start' in item and 'end' in item]
        unused_saved_lines=list(saved_lines)
        detected=detect_pink_obstacle_geometry(image)
        unused_detected_circles=list(detected.circles)
        unused_detected_lines=list(detected.lines)
        for b in boxes:
            if b.class_id==4:
                cx,cy=b.center_x*w,b.center_y*h
                if unused_saved_lines:
                    chosen=min(unused_saved_lines,key=lambda x:hypot((x.start[0]+x.end[0])/2-cx,(x.start[1]+x.end[1])/2-cy))
                    unused_saved_lines.remove(chosen); lines.append(chosen)
                else:
                    color_candidates=sorted(
                        unused_detected_lines,
                        key=lambda x:hypot((x.start[0]+x.end[0])/2-cx,(x.start[1]+x.end[1])/2-cy),
                    )
                    pose=pose_for('obstacle_line',b)
                    pose_points=getattr(pose,'keypoints',()) if pose else ()
                    pose_candidates=[]
                    if len(pose_points)>=2 and pose_points[0].visible>0 and pose_points[1].visible>0:
                        pose_candidates=[LineAnnotation(4,(pose_points[0].x,pose_points[0].y),(pose_points[1].x,pose_points[1].y))]
                    box_candidate=_line_from_box(b,w,h)
                    chosen=_select_color_geometry(
                        [LineAnnotation(4,item.start,item.end) for item in color_candidates],
                        pose_candidates,
                        box_candidate,
                    )
                    if color_candidates:
                        unused_detected_lines.remove(color_candidates[0])
                    lines.append(chosen)
            elif b.class_id != 1:
                box_center=(b.center_x*w,b.center_y*h)
                box_radius=max(b.width*w,b.height*h)/2
                already_saved=any(
                    a.class_id == b.class_id
                    and hypot(a.center_x-box_center[0],a.center_y-box_center[1]) <= a.radius + box_radius
                    for a in saved_circles
                )
                if not already_saved:
                    color_candidates=sorted(
                        unused_detected_circles,
                        key=lambda item:hypot(item.center[0]-box_center[0],item.center[1]-box_center[1]),
                    )
                    if b.class_id == 3 and color_candidates:
                        chosen=color_candidates[0]
                        circles.append(CircleAnnotation(3,chosen.center[0],chosen.center[1],chosen.radius))
                        unused_detected_circles.remove(chosen)
                    else:
                        circles.append(CircleAnnotation(b.class_id,box_center[0],box_center[1],box_radius))
        lines = _merge_line_annotations(lines)
        center=None; muzzle=None; pending=None; selected_object=None; window='ShellShock Pose annotation'; dw,dh=_display_size(w,h)
        label_center = next(((box.center_x*w, box.center_y*h) for box in boxes if box.class_id == 1), None)
        zoom=1.; view_cx,view_cy=w/2.,h/2.; barrel_px=dynamic_barrel_length(w,barrel_length)
        if geometry_path.exists():
            try:
                if saved_geometry.get('self_center'):
                    center=tuple(saved_geometry['self_center'])
                if saved_geometry.get('self_muzzle'):
                    muzzle=tuple(saved_geometry['self_muzzle'])
            except (OSError, ValueError, TypeError):
                muzzle=None
        if center is None:
            pose=next((item for item in pose_detections if item.name=='self' and item.keypoints and item.keypoints[0].visible>0),None)
            if pose: center=(pose.keypoints[0].x,pose.keypoints[0].y)
        if center is None and label_center is not None:
            center=label_center
        solver_points=(); solver_result={}; solver_signature=None; solver_mode="normal_low"
        def current_signature():
            return repr((circles,lines,center,wind_signed,solver_mode))
        def recompute_shot():
            nonlocal solver_points,solver_result,solver_signature
            scene=scene_from_editor(circles,lines,w,h,center,muzzle,{**dict(meta or {}),"wind_value":abs(wind_signed),"wind_direction":"left" if wind_signed<0 else "right"})
            analysis=analyze_frame(image,scene=scene)
            enemies=[box.center for box in scene.boxes if box.name=="enemy"]
            if analysis.world.self_position is None or not enemies:
                solver_result={"status":"unreachable","reason":"mark-self-and-enemy"};solver_points=()
            else:
                solver_result=solve_integer_shot(analysis.world.self_position,enemies[0],analysis.world,analysis.wind_value,analysis.wind_direction,w,solver_mode,barrel_extension=barrel_length)
                solver_points=sample_solution_trajectory(solver_result,solver_result.get("launch_point",center),w,analysis.wind_value,analysis.wind_direction,world=analysis.world)
            solver_signature=current_signature()
            print("SOLVER",solver_mode,solver_result.get("power"),solver_result.get("angle_degrees"),solver_result.get("reason",""),flush=True)
            redraw()
        def image_point(x,y):
            # The rendered crop has width w/zoom and height h/zoom.
            # Therefore display pixels per source pixel grow with zoom.
            scale_x=dw*zoom/w; scale_y=dh*zoom/h
            return (view_cx+(x-dw/2.)/scale_x, view_cy+(y-dh/2.)/scale_y)
        def redraw():
            nonlocal view_cx,view_cy
            rendered=_draw(image,circles,lines,center,muzzle,square,pose_detections)
            if solver_signature==current_signature():
                draw_trajectory(rendered,solver_points)
            crop_w=min(w,max(1,int(w/zoom))); crop_h=min(h,max(1,int(h/zoom)))
            left=int(round(view_cx-crop_w/2)); top=int(round(view_cy-crop_h/2))
            left=max(0,min(w-crop_w,left)); top=max(0,min(h-crop_h,top)); view_cx=left+crop_w/2.; view_cy=top+crop_h/2.
            view=cv2.resize(rendered[top:top+crop_h,left:left+crop_w],(dw,dh),interpolation=cv2.INTER_AREA if zoom<=1 else cv2.INTER_LINEAR)
            angle_text='N/A' if angle is None else f'{angle:.1f}deg'
            direction_text=str((meta or {}).get('direction','right')).upper()
            wind_text=f'{abs(wind_signed):.0f} {"LEFT" if wind_signed<0 else "RIGHT"}'
            cv2.putText(view,f'{index+1}/{len(images)} {path.name} | {selected}: {CLASS_LABELS[selected]} | ANGLE={angle_text} {direction_text} | WIND={wind_text} | RADIUS={radius:.1f}px | BARREL={barrel_px:.1f}px | ZOOM={zoom:.2f}x | A/D 1deg Q/E 5deg Z/C wind F5 solve T/H/R mode U arc',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),2); cv2.imshow(window,view)
        def adjust_angle(side, amount):
            nonlocal angle,meta
            angle=0. if angle is None else angle
            direction=str((meta or {}).get('direction','right')).lower()
            if side=='right':
                if direction=='right':
                    angle += amount
                    if angle>90.: angle=180.-angle; direction='left'
                else:
                    angle -= amount
                    if angle<0.: angle=-angle; direction='right'
            else:
                if direction=='left':
                    angle += amount
                    if angle>90.: angle=180.-angle; direction='right'
                else:
                    angle -= amount
                    if angle<0.: angle=-angle; direction='left'
            meta=dict(meta or {})
            meta['direction']=direction
            meta=dict(meta or {})
            meta['angle_degrees']=angle
            _save_metadata_angle(metadata_path,meta,angle)
            redraw()
        def adjust_wind(delta):
            nonlocal wind_signed,meta
            wind_signed=max(-100.,min(100.,wind_signed+delta))
            meta=dict(meta or {})
            meta['wind_value']=round(abs(wind_signed),2)
            meta['wind_direction']='left' if wind_signed<0 else 'right'
            meta['wind_source']='manual'
            _save_metadata_angle(metadata_path,meta,angle)
            _save_wind_label(wind_label_dir,wind_crops_dir,path.stem,wind_signed,'manual')
            redraw()
        def mouse(event,x,y,_flags,_userdata):
            nonlocal pending,center,muzzle,circles,lines,selected_object,zoom,view_cx,view_cy
            px=tuple(round(v) for v in image_point(x,y))
            px=(max(0,min(w-1,px[0])),max(0,min(h-1,px[1])))
            if event==cv2.EVENT_MOUSEWHEEL:
                old_zoom=zoom; direction=_wheel_delta(_flags)
                if direction==0: return
                anchor=image_point(x,y)
                zoom=max(1.,min(8.,zoom*(1.25 if direction>0 else 0.8)))
                if zoom!=old_zoom:
                    view_cx=anchor[0]-(x-dw/2.)/(dw*zoom/w)
                    view_cy=anchor[1]-(y-dh/2.)/(dh*zoom/h)
                    redraw()
                return
            if event!=cv2.EVENT_LBUTTONDOWN and event!=cv2.EVENT_RBUTTONDOWN:return
            if event==cv2.EVENT_RBUTTONDOWN:
                if selected==1 and center is not None and hypot(center[0]-px[0],center[1]-px[1])<=max(16.,12./zoom):
                    center=None; selected_object=None; redraw(); return
                delete_class=2 if selected==1 else selected
                before=len(circles); circles=delete_nearest_annotation(circles,px[0],px[1],delete_class)
                if len(circles)!=before:
                    selected_object=None
                elif selected==4:
                    before=len(lines); lines=delete_nearest_line(lines,px[0],px[1])
                    if len(lines)!=before: selected_object=None
                redraw(); return
            if event==cv2.EVENT_LBUTTONDOWN:
                center_action=_center_click_action(selected,center,circles,px[0],px[1])
                if center_action=='select_center':
                    selected_object=('center',None); redraw(); return
                if center_action=='set_center':
                    center=px; selected_object=('center',None); redraw(); return
            hit=_circle_at(circles,px[0],px[1],2 if selected==1 else selected)
            if event==cv2.EVENT_LBUTTONDOWN and hit is not None:
                selected_object=('circle',hit); redraw(); return
            if selected==4:
                line_hits=[(i,_point_segment_distance(px[0],px[1],a.start,a.end)) for i,a in enumerate(lines)]
                line_hits=[item for item in line_hits if item[1]<=max(14.,8./zoom)]
                if line_hits:
                    selected_object=('line',min(line_hits,key=lambda item:item[1])[0]); redraw(); return
                if pending is None: pending=px
                else: lines.append(LineAnnotation(4,pending,px)); selected_object=('line',len(lines)-1); pending=None
            elif selected==2:
                circles=[a for a in circles if a.class_id!=2]; circles.append(CircleAnnotation(2,px[0],px[1],max(8,radius*2))); selected_object=('circle',len(circles)-1)
            else: circles.append(CircleAnnotation(selected,px[0],px[1],radius)); selected_object=('circle',len(circles)-1)
            redraw()
        cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window,dw,dh)
        cv2.setMouseCallback(window,mouse); redraw()
        while True:
            key=cv2.waitKeyEx(30)
            if key==7602176:
                recompute_shot(); continue
            if key in (ord("r"),ord("h"),ord("t")):
                solver_mode=select_mode(chr(key),solver_mode);recompute_shot();continue
            if key==ord("u"):
                solver_mode=select_mode("page down" if solver_mode.endswith("high") else "page up",solver_mode);recompute_shot();continue
            if key in (ord('+'),ord('=')):
                if selected_object and selected_object[0]=='circle':
                    circles,selected_object=_adjust_circle_radius(circles,selected_object,2)
                else: radius=min(1000,radius+2)
                redraw(); continue
            if key in (ord('-'),ord('_')):
                if selected_object and selected_object[0]=='circle':
                    circles,selected_object=_adjust_circle_radius(circles,selected_object,-2)
                else: radius=max(4,radius-2)
                redraw(); continue
            key_char=key & 0xff
            if key_char in (ord('d'),ord('D')):
                adjust_angle('left',1.); continue
            if key_char in (ord('a'),ord('A')):
                adjust_angle('right',1.); continue
            if key_char in (ord('q'),ord('Q')):
                adjust_angle('right',5.); continue
            if key_char in (ord('e'),ord('E')):
                adjust_angle('left',5.); continue
            if key_char in (ord('z'),ord('Z')):
                adjust_wind(-1.); continue
            if key_char in (ord('c'),ord('C')):
                adjust_wind(1.); continue
            if key in (81,82,83,84,2424832,2490368,2555904,2621440):
                if selected_object:
                    dx,dy={81:(-1,0),82:(0,-1),83:(1,0),84:(0,1),2424832:(-1,0),2490368:(0,-1),2555904:(1,0),2621440:(0,1)}[key]
                    if selected_object[0]=='circle':
                        i=selected_object[1]; a=circles[i]; circles[i]=CircleAnnotation(a.class_id,a.center_x+dx,a.center_y+dy,a.radius)
                    elif selected_object[0]=='line':
                        i=selected_object[1]; a=lines[i]; lines[i]=LineAnnotation(4,(a.start[0]+dx,a.start[1]+dy),(a.end[0]+dx,a.end[1]+dy))
                    elif selected_object[0]=='center': center=(center[0]+dx,center[1]+dy)
                    elif selected_object[0]=='muzzle': muzzle=(muzzle[0]+dx,muzzle[1]+dy)
                    redraw(); continue
            if 48<=key<=57: selected=key-48; redraw(); continue
            if key in (ord('b'),ord('B')): square=not square; redraw(); continue
            action=key_action(key)
            if action is None: continue
            preview_path=Path(preview_dir)/f'{path.stem}.jpg'
            _save_state(override,geometry_path,circles,lines,w,h,center,muzzle); _save_metadata_angle(metadata_path,meta,angle); _save_wind_label(wind_label_dir,wind_crops_dir,path.stem,wind_signed,'manual' if meta.get('wind_source')=='manual' else 'detected'); Path(preview_dir).mkdir(parents=True,exist_ok=True); cv2.imwrite(str(preview_path),_draw(image,circles,lines,center,muzzle,square,pose_detections))
            if action=='accept':
                move_annotation_bundle(path,override,geometry_path,metadata_path,preview_path,Path(wind_label_dir)/f'{path.stem}.json',annotation_dir)
                images.pop(index)
                if not images: cv2.destroyAllWindows(); return
                index=min(index,len(images)-1)
                break
            if action=='quit': cv2.destroyAllWindows(); return
            index+=-1 if action=='previous' else 1; break
    cv2.destroyAllWindows()


def run_merged_annotation(images, raw_dir, override_dir, preview_dir, geometry_dir,
                          metadata_dir, barrel_length, wind_label_dir, wind_crops_dir,
                          annotation_dir, weights=None, confidence=.35):
    """Run annotation and replay in one window, toggled by CapsLock."""
    if not images:
        raise ValueError("no images matched the selected timestamp range")
    interaction_mode = ANNOTATION_MODE
    requested_toggle = False
    requested_alt = False
    hooks = []

    def request_toggle(_event=None):
        nonlocal requested_toggle
        requested_toggle = True

    def request_alt(_event=None):
        nonlocal requested_alt
        requested_alt = True

    hooks.extend((keyboard.on_press_key("caps lock", request_toggle), keyboard.on_press_key("alt", request_alt)))
    index, selected, radius, square = 0, 0, 24.0, False
    pose_detector = YoloDetector(str(weights), confidence) if weights else None

    try:
        while 0 <= index < len(images):
            path = images[index]
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"unable to read image: {path}")
            h, w = image.shape[:2]
            override = Path(override_dir) / f"{path.stem}.txt"
            geometry_path = Path(geometry_dir) / f"{path.stem}.json"
            metadata_path = Path(metadata_dir) / f"{path.stem}.json"
            meta = dict(_metadata(metadata_path) or {})
            wind_value = float(meta.get("wind_value", 0))
            wind_direction = str(meta.get("wind_direction", "right")).lower()
            wind_signed = abs(wind_value) * (-1.0 if wind_direction == "left" else 1.0)
            meta["wind_value"] = round(abs(wind_signed), 2)
            meta["wind_direction"] = "left" if wind_signed < 0 else "right"
            meta["wind_source"] = str(meta.get("wind_source", "manual"))
            boxes = _merge_duplicate_line_boxes(_load_boxes(override if override.exists() else Path(raw_dir) / f"{path.stem}.txt", w, h))
            circles, lines = [], []
            pose_detections = pose_detector.detect(image) if pose_detector else []
            saved_geometry = {}
            if geometry_path.exists():
                try:
                    saved_geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    pass
            circles = [
                CircleAnnotation(int(item["class_id"]), float(item["center"][0]), float(item["center"][1]), float(item["radius"]))
                for item in saved_geometry.get("circles", ())
                if isinstance(item, dict) and "class_id" in item and "center" in item and len(item["center"]) == 2 and "radius" in item
            ]
            saved_lines = [
                LineAnnotation(4, tuple(item["start"]), tuple(item["end"]))
                for item in saved_geometry.get("lines", ())
                if isinstance(item, dict) and "start" in item and "end" in item
            ]
            lines = []
            detected = detect_pink_obstacle_geometry(image)
            unused_circles, unused_lines = list(detected.circles), list(detected.lines)
            for box in boxes:
                cx, cy = box.center_x * w, box.center_y * h
                if box.class_id == 4:
                    if saved_lines:
                        chosen = saved_lines.pop(0)
                    else:
                        candidates = sorted(unused_lines, key=lambda item: hypot((item.start[0] + item.end[0]) / 2 - cx, (item.start[1] + item.end[1]) / 2 - cy))
                        chosen = LineAnnotation(4, candidates[0].start, candidates[0].end) if candidates else _line_from_box(box, w, h)
                        if candidates:
                            unused_lines.remove(candidates[0])
                    lines.append(LineAnnotation(4, chosen.start, chosen.end))
                elif box.class_id != 1 and not any(a.class_id == box.class_id and hypot(a.center_x - cx, a.center_y - cy) <= a.radius + max(box.width * w, box.height * h) / 2 for a in circles):
                    box_radius = max(box.width * w, box.height * h) / 2
                    if box.class_id == 3 and unused_circles:
                        chosen = min(unused_circles, key=lambda item: hypot(item.center[0] - cx, item.center[1] - cy))
                        unused_circles.remove(chosen)
                        circles.append(CircleAnnotation(3, chosen.center[0], chosen.center[1], chosen.radius))
                    else:
                        circles.append(CircleAnnotation(box.class_id, cx, cy, box_radius))
            lines = _merge_line_annotations(lines)
            center = tuple(saved_geometry["self_center"]) if saved_geometry.get("self_center") else None
            muzzle = tuple(saved_geometry["self_muzzle"]) if saved_geometry.get("self_muzzle") else None
            if center is None:
                pose = next((item for item in pose_detections if item.name == "self" and item.keypoints and item.keypoints[0].visible > 0), None)
                center = (pose.keypoints[0].x, pose.keypoints[0].y) if pose else None
            if center is None:
                center = next(((box.center_x * w, box.center_y * h) for box in boxes if box.class_id == 1), None)
            label_center = next(((box.center_x * w, box.center_y * h) for box in boxes if box.class_id == 1), None)
            selected_object = None
            pending = None
            zoom, view_cx, view_cy = 1.0, w / 2.0, h / 2.0
            dw, dh = _display_size(w, h)
            window = "ShellShock Annotation / Replay"
            solver_mode = "normal_low"
            candidate_index = 0
            report, replay_overlay = {}, image
            annotation_points = ()
            replay_log_cache = []
            save_status = ""
            save_status_until = 0.0
            replay_target = None
            show_annotations, show_guide, manual_adjust = True, True, False
            angle_override = power_override = None
            applied_wind_value = None
            wind_value_control = wind_signed
            cursor = [center[0] if center else w / 2, center[1] if center else h / 2]

            def current_scene():
                return scene_from_editor(circles, lines, w, h, center, muzzle, {**meta, "wind_value": abs(wind_signed), "wind_direction": "left" if wind_signed < 0 else "right"})

            def save_current():
                scene = current_scene()
                save_manual_scene(scene, override, geometry_path, metadata_path, backup=False)
                _save_wind_label(wind_label_dir, wind_crops_dir, path.stem, wind_signed, "manual" if meta.get("wind_source") == "manual" else "detected")

            def recalculate_annotation():
                nonlocal annotation_points
                scene = current_scene()
                analysis = analyze_frame(image, scene=scene)
                enemies = [box.center for box in scene.boxes if box.name == "enemy"]
                if analysis.world.self_position is None or not enemies:
                    annotation_points = ()
                    return
                result = solve_integer_shot(
                    analysis.world.self_position, enemies[0], analysis.world,
                    analysis.wind_value, analysis.wind_direction, w, solver_mode,
                    barrel_extension=barrel_length,
                )
                annotation_points = sample_solution_trajectory(
                    result, result.get("launch_point", center), w,
                    analysis.wind_value, analysis.wind_direction, world=analysis.world,
                )

            def recalculate(trigger=None):
                nonlocal report, replay_overlay, applied_wind_value
                scene = current_scene()
                report, replay_overlay, _ignored, _paths = replay_image(
                    path, source="annotation", weights=weights, mode=solver_mode, root=Path(raw_dir).parent.parent,
                    output_root=Path(annotation_dir) / "replay", target=replay_target, scene_override=scene,
                    barrel_extension=barrel_length, angle_override=angle_override, power_override=power_override,
                    wind_override=applied_wind_value, persist=False, candidate_index=candidate_index,
                )
                report["pending_wind_value"] = wind_value_control
                if trigger:
                    replay_log_cache.append({"timestamp": datetime.now().isoformat(timespec="seconds"), "trigger": trigger, "report": dict(report)})

            def save_replay_log_cache():
                nonlocal save_status, save_status_until
                destination = Path(annotation_dir) / "replay" / "logs" / f"{path.stem}.log"
                save_status = save_replay_log_bundle(destination, replay_log_cache)
                save_status_until = monotonic() + 3.0

            def image_point(x, y):
                scale_x, scale_y = dw * zoom / w, dh * zoom / h
                return (view_cx + (x - dw / 2) / scale_x, view_cy + (y - dh / 2) / scale_y)

            def redraw():
                nonlocal view_cx, view_cy
                if interaction_mode == ANNOTATION_MODE:
                    rendered = _draw(image, circles, lines, center, muzzle, square, pose_detections)
                    if annotation_points:
                        draw_trajectory(rendered, annotation_points)
                    status = f"ANNOTATION | {selected}: {CLASS_LABELS[selected]} | WIND={format_wind_text(wind_signed)} | RADIUS={radius:.1f}px | Z/C wind | +/- resize | CAPSLOCK replay"
                else:
                    rendered = render_replay_visual(image, report.get("predicted_points", ()), guide_from_report(report), current_scene(), report, show_annotations=show_annotations, show_guide=show_guide)
                    current_solution = report.get("solution", {})
                    candidate_text = f" | CANDIDATE={current_solution.get('candidate_index', 0) + 1}/{current_solution.get('candidate_count')}" if current_solution.get("candidate_count") else ""
                    status = f"REPLAY | MODE={solver_mode}{candidate_text} | ANGLE={current_solution.get('angle_degrees', 'N/A')} POWER={current_solution.get('power', 'N/A')} | {'TUNING' if manual_adjust else 'TAB tuning'} | ALT aim | S save logs | CAPSLOCK annotation"
                crop_w, crop_h = min(w, max(1, int(w / zoom))), min(h, max(1, int(h / zoom)))
                left = max(0, min(w - crop_w, int(round(view_cx - crop_w / 2))))
                top = max(0, min(h - crop_h, int(round(view_cy - crop_h / 2))))
                view_cx, view_cy = left + crop_w / 2, top + crop_h / 2
                view = cv2.resize(rendered[top:top + crop_h, left:left + crop_w], (dw, dh), interpolation=cv2.INTER_AREA if zoom <= 1 else cv2.INTER_LINEAR)
                cv2.putText(view, f"{index + 1}/{len(images)} {path.name} | {status}", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 2)
                if save_status and monotonic() < save_status_until:
                    cv2.putText(view, save_status, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 180), 2)
                cv2.imshow(window, view)

            def mouse(event, x, y, flags, _userdata):
                nonlocal pending, center, circles, lines, selected_object, zoom, view_cx, view_cy, replay_target
                px = tuple(round(v) for v in image_point(x, y))
                px = (max(0, min(w - 1, px[0])), max(0, min(h - 1, px[1])))
                cursor[0], cursor[1] = px
                if event == cv2.EVENT_MOUSEWHEEL:
                    direction = _wheel_delta(flags)
                    if direction:
                        anchor = image_point(x, y)
                        zoom = max(1., min(8., zoom * (1.25 if direction > 0 else .8)))
                        view_cx = anchor[0] - (x - dw / 2) / (dw * zoom / w)
                        view_cy = anchor[1] - (y - dh / 2) / (dh * zoom / h)
                        redraw()
                    return
                if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                    return
                if interaction_mode == REPLAY_MODE:
                    if event == cv2.EVENT_LBUTTONDOWN:
                        hits = [(i, (item.center[0] - px[0]) ** 2 + (item.center[1] - px[1]) ** 2) for i, item in enumerate(current_scene().boxes)]
                        selected_object = min(hits, key=lambda item: item[1])[0] if hits and min(hits, key=lambda item: item[1])[1] <= 50 ** 2 else None
                    return
                if event == cv2.EVENT_RBUTTONDOWN:
                    if selected == 1 and center is not None and hypot(center[0] - px[0], center[1] - px[1]) <= max(16., 12. / zoom):
                        center, selected_object = None, None
                    else:
                        before = len(circles); circles = delete_nearest_annotation(circles, px[0], px[1], 2 if selected == 1 else selected)
                        if len(circles) == before and selected == 4:
                            lines = delete_nearest_line(lines, px[0], px[1])
                        selected_object = None
                    redraw(); return
                center_action = _center_click_action(selected, center, circles, px[0], px[1])
                if center_action == "select_center":
                    selected_object = ("center", None); redraw(); return
                if center_action == "set_center":
                    center = px; selected_object = ("center", None); redraw(); return
                hit = _circle_at(circles, px[0], px[1], 2 if selected == 1 else selected)
                if hit is not None:
                    selected_object = ("circle", hit); redraw(); return
                if selected == 4:
                    line_hits = [(i, _point_segment_distance(px[0], px[1], item.start, item.end)) for i, item in enumerate(lines)]
                    line_hits = [item for item in line_hits if item[1] <= max(14., 8. / zoom)]
                    if line_hits:
                        selected_object = ("line", min(line_hits, key=lambda item: item[1])[0]); redraw(); return
                    if pending is None: pending = px
                    else: lines.append(LineAnnotation(4, pending, px)); selected_object = ("line", len(lines) - 1); pending = None
                elif selected == 2:
                    circles = [a for a in circles if a.class_id != 2]; circles.append(CircleAnnotation(2, px[0], px[1], max(8, radius * 2))); selected_object = ("circle", len(circles) - 1)
                else:
                    circles.append(CircleAnnotation(selected, px[0], px[1], radius)); selected_object = ("circle", len(circles) - 1)
                redraw()

            def adjust_wind(delta):
                nonlocal wind_signed, wind_value_control, meta
                wind_signed = max(-100., min(100., wind_signed + delta))
                wind_value_control = wind_signed
                meta = annotation_wind_metadata(meta, wind_signed)
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                _save_wind_label(wind_label_dir, wind_crops_dir, path.stem, wind_signed, "manual")
                redraw()

            cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(window, dw, dh)
            cv2.setMouseCallback(window, mouse)
            if interaction_mode == REPLAY_MODE:
                recalculate()
            redraw()
            while True:
                if requested_toggle:
                    requested_toggle = False
                    save_current()
                    interaction_mode = toggle_interaction_mode(interaction_mode)
                    if interaction_mode == REPLAY_MODE:
                        recalculate()
                    redraw(); continue
                if requested_alt:
                    requested_alt = False
                    if interaction_mode == REPLAY_MODE:
                        replay_target = tuple(cursor); candidate_index = 0; recalculate("ALT"); redraw()
                    continue
                key = cv2.waitKeyEx(30)
                key_char = key & 0xff
                if key_char in (ord(","), ord("<"), ord("."), ord(">")):
                    save_current()
                    index = cycle_image_index(index, -1 if key_char in (ord(","), ord("<")) else 1, len(images))
                    break
                if key == 7602176:
                    if interaction_mode == REPLAY_MODE: recalculate()
                    else: recalculate_annotation()
                    redraw(); continue
                action = interaction_key_action(interaction_mode, key)
                if action and action[0] == "select_class": selected = action[1]; redraw(); continue
                if action and action[0] == "resize":
                    circles, selected_object = _adjust_circle_radius(circles, selected_object, action[1]) if selected_object and selected_object[0] == "circle" else (circles, selected_object)
                    if not selected_object or selected_object[0] != "circle": radius = max(4, min(1000, radius + action[1]))
                    redraw(); continue
                if interaction_mode == REPLAY_MODE:
                    if action and action[0] == "solver_mode": candidate_index = 0; solver_mode = action[1]; recalculate(); redraw(); continue
                    if action and action[0] == "toggle_annotations": show_annotations = not show_annotations; redraw(); continue
                    if action and action[0] == "toggle_tuning":
                        manual_adjust = not manual_adjust
                        if manual_adjust:
                            tuning_scene = current_scene()
                            angle_override, power_override = manual_controls(tuning_scene, report.get("solution", {}))
                            meta["direction"] = tuning_scene.metadata.get("direction", meta.get("direction", "right"))
                        else:
                            angle_override = power_override = applied_wind_value = None
                        recalculate(); redraw(); continue
                    if manual_adjust and wind_adjustment(key & 0xff) is not None:
                        wind_value_control += wind_adjustment(key & 0xff); report["pending_wind_value"] = wind_value_control; continue
                    if manual_adjust and key in (10, 13):
                        applied_wind_value = wind_value_control; recalculate(); redraw(); continue
                    if manual_adjust and arrow_adjustment(key) is not None:
                        field, delta = arrow_adjustment(key)
                        if field == "angle": angle_override = max(0., min(90., angle_override + delta))
                        else: power_override = max(1, min(100, power_override + delta))
                        recalculate(); redraw(); continue
                    if key in (2162688, 2228224):
                        candidates = report.get("solution", {}).get("candidate_count", 0)
                        if solver_mode.startswith("reflection_") and candidates:
                            candidate_index = cycle_image_index(candidate_index, 1 if key == 2228224 else -1, candidates)
                        else:
                            candidate_index = 0
                            solver_mode = select_mode("page up" if key == 2162688 else "page down", solver_mode)
                        recalculate(); redraw(); continue
                    if key in (ord("r"), ord("R"), ord("h"), ord("H"), ord("t"), ord("T")):
                        solver_mode = select_mode(chr(key).lower(), solver_mode); recalculate(); redraw(); continue
                    if key in (ord("e"), ord("E")):
                        replay_target = tuple(cursor); candidate_index = 0; recalculate(); redraw(); continue
                    if key in (ord("g"), ord("G")):
                        show_guide = not show_guide; redraw(); continue
                    if key in (ord("f"), ord("F")):
                        full = cv2.getWindowProperty(window, cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if full > 0 else cv2.WINDOW_FULLSCREEN)
                        continue
                    if key in (ord("s"), ord("S")):
                        save_replay_log_cache(); continue
                    if key in (ord("q"), ord("Q")):
                        return
                    if key == 27:
                        return
                    if key in (ord("<"), ord(","), ord(">"), ord(".")):
                        save_current(); index = cycle_image_index(index, -1 if key in (ord("<"), ord(",")) else 1, len(images)); break
                else:
                    if key in (ord("r"), ord("R"), ord("h"), ord("H"), ord("t"), ord("T")):
                        solver_mode = select_mode(chr(key).lower(), solver_mode); recalculate_annotation(); redraw(); continue
                    if key == ord("u"):
                        solver_mode = select_mode("page down" if solver_mode.endswith("high") else "page up", solver_mode)
                        recalculate_annotation(); redraw(); continue
                    if key & 0xff in (ord("z"), ord("Z")):
                        adjust_wind(-1.); continue
                    if key & 0xff in (ord("c"), ord("C")):
                        adjust_wind(1.); continue
                    if key in (81, 82, 83, 84, 2424832, 2490368, 2555904, 2621440) and selected_object:
                        dx, dy = {81: (-1, 0), 82: (0, -1), 83: (1, 0), 84: (0, 1), 2424832: (-1, 0), 2490368: (0, -1), 2555904: (1, 0), 2621440: (0, 1)}[key]
                        if selected_object[0] == "circle":
                            i = selected_object[1]; item = circles[i]
                            circles[i] = CircleAnnotation(item.class_id, item.center_x + dx, item.center_y + dy, item.radius)
                        elif selected_object[0] == "line":
                            i = selected_object[1]; item = lines[i]
                            lines[i] = LineAnnotation(4, (item.start[0] + dx, item.start[1] + dy), (item.end[0] + dx, item.end[1] + dy))
                        elif selected_object[0] == "center" and center is not None:
                            center = (center[0] + dx, center[1] + dy)
                        redraw(); continue
                    if key in (ord("b"), ord("B")):
                        square = not square; redraw(); continue
                if key in (ord(","), ord("<"), ord("."), ord(">")):
                    save_current()
                    index = cycle_image_index(index, -1 if key in (ord(","), ord("<")) else 1, len(images))
                    break
                if interaction_mode == REPLAY_MODE:
                    continue
                if key in (2162688, 2228224):
                    continue
                common = key_action(key)
                if common is None: continue
                save_current()
                preview_path = Path(preview_dir) / f"{path.stem}.jpg"
                Path(preview_dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(preview_path), _draw(image, circles, lines, center, muzzle, square, pose_detections))
                if common == "accept":
                    move_annotation_bundle(path, override, geometry_path, metadata_path, preview_path, Path(wind_label_dir) / f"{path.stem}.json", annotation_dir)
                    images.pop(index)
                    if not images: return
                    index = min(index, len(images) - 1); break
                if common == "quit": return
                index += -1 if common == "previous" else 1
                break
            cv2.destroyWindow(window)
    finally:
        for hook in hooks:
            keyboard.unhook(hook)
        cv2.destroyAllWindows()

def write_enemy_supplement(supplemental_dir,stem,points,image_width,image_height):
    p=Path(supplemental_dir)/f'{stem}.txt'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('\n'.join(yolo_label_line(0,x,image_width,image_height) for x in points)+'\n',encoding='utf-8'); return p
def build_parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--raw-dir',type=Path,default=(DATA_ROOT / 'yolo_captures/full')); p.add_argument('--override-dir',type=Path,default=(DATA_ROOT / 'yolo_captures/labels')); p.add_argument('--geometry-dir',type=Path,default=(DATA_ROOT / 'yolo_captures/pose_geometry')); p.add_argument('--metadata-dir',type=Path,default=(DATA_ROOT / 'yolo_captures/metadata')); p.add_argument('--wind-label-dir',type=Path,default=(DATA_ROOT / 'yolo_captures/wind/labels')); p.add_argument('--wind-crops-dir',type=Path,default=(DATA_ROOT / 'yolo_captures/wind')); p.add_argument('--preview-dir',type=Path,default=(DATA_ROOT / 'yolo_captures/previews')); p.add_argument('--annotation-dir',type=Path,default=(DATA_ROOT / 'annotate')); p.add_argument('--weights',type=Path,default=(DATA_ROOT / 'runs/shellshock_yolo11n_pose_v1/weights/best.pt')); p.add_argument('--confidence',type=float,default=.35); p.add_argument('--no-yolo-prelabel',action='store_true'); p.add_argument('--overwrite-yolo-labels',action='store_true'); p.add_argument('--barrel-length',type=float,default=35.); p.add_argument('--start',default=''); p.add_argument('--end',default='\U0010ffff'); p.add_argument('--all-images',action='store_true'); return p
def main():
    a=build_parser().parse_args()
    images=select_images(a.raw_dir,'','\U0010ffff') if a.all_images else select_images(a.raw_dir,a.start,a.end)
    if not a.no_yolo_prelabel:
        from prelabel_yolo import run as run_yolo_prelabel
        run_yolo_prelabel(a.weights,a.raw_dir,a.override_dir,a.confidence,a.overwrite_yolo_labels)
        images=select_images(a.raw_dir,'','\U0010ffff') if a.all_images else select_images(a.raw_dir,a.start,a.end)
    run_merged_annotation(images,a.raw_dir,a.override_dir,a.preview_dir,a.geometry_dir,a.metadata_dir,a.barrel_length,a.wind_label_dir,a.wind_crops_dir,a.annotation_dir,a.weights,a.confidence)
if __name__=='__main__': main()
