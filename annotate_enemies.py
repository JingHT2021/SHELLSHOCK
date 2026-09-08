"""Interactive multi-class circle/line annotation editor."""
from __future__ import annotations

import argparse, json
from dataclasses import dataclass
from math import cos, hypot, radians, sin
from pathlib import Path
import cv2

from shellshock_detector.training_data import yolo_label_line
from shellshock_detector.yolo_dataset import CLASS_NAMES, YoloBox, parse_yolo_label_text
from shellshock_detector.obstacle_geometry import detect_pink_obstacle_geometry
from shellshock_detector.wind import detect_wind

CLASS_KEY_MAP = {str(i): i for i in range(10)}
CLASS_LABELS = {0:"enemy / 敌人",1:"self_muzzle / 炮管终点",2:"self / 己方中心",3:"obstacle_circle / 圆形障碍物",4:"obstacle_line / 线段障碍物",5:"portal_orange / 橙色虫洞",6:"portal_blue / 蓝色虫洞",7:"blackhole / 黑洞",8:"double_damage / 二倍伤害",9:"Triple_damage / 三倍伤害"}
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

def select_images(raw_dir: Path, start: str, end: str):
    return sorted((p for p in Path(raw_dir).iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.webp'} and start<=p.stem<=end),key=lambda p:p.name)

def display_to_image_point(point, display_size, image_size): return round(point[0]*image_size[0]/display_size[0]),round(point[1]*image_size[1]/display_size[1])
def key_action(key):
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

def _save_wind_label(label_dir, wind_crops_dir, stem, wind_signed):
    crop_path=Path(wind_crops_dir)/f'{stem}.png'
    if not crop_path.exists(): return
    signed=round(float(wind_signed),2)
    payload={'image':str(crop_path), 'stem':stem, 'wind_value':abs(signed), 'wind_direction':'left' if signed<0 else 'right', 'wind_signed':signed}
    path=Path(label_dir)/f'{stem}.json'; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def _line_from_box(box,w,h):
    cx,cy=box.center_x*w,box.center_y*h; width,height=box.width*w,box.height*h
    if width>=height: return LineAnnotation(4,(cx-width/2,cy),(cx+width/2,cy))
    return LineAnnotation(4,(cx,cy-height/2),(cx,cy+height/2))

def _save_state(override_path, geometry_path, circles, lines, width, height, muzzle):
    labels=[circle_to_yolo_box(a,width,height) for a in circles]+[line_to_yolo_box(a,width,height) for a in lines]
    override_path.parent.mkdir(parents=True,exist_ok=True); override_path.write_text('\n'.join(labels)+ ('\n' if labels else ''),encoding='utf-8')
    geometry_path.parent.mkdir(parents=True,exist_ok=True)
    geometry_path.write_text(json.dumps({'lines':[{'start':list(a.start),'end':list(a.end)} for a in lines], 'self_muzzle':list(muzzle) if muzzle else None},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def _draw(image,circles,lines,muzzle,square=False):
    out=image.copy(); colors=[(0,0,255),(0,200,0),(255,0,0),(0,200,255),(255,255,0),(0,140,255),(255,120,0),(180,0,255),(0,255,180),(255,0,180)]
    for a in circles:
        c=colors[a.class_id%len(colors)]; p=(round(a.center_x),round(a.center_y))
        display_radius=a.radius*SELF_DISPLAY_SCALE if a.class_id==2 else a.radius
        if square: cv2.rectangle(out,(round(a.center_x-display_radius),round(a.center_y-display_radius)),(round(a.center_x+display_radius),round(a.center_y+display_radius)),c,3)
        else: cv2.circle(out,p,max(1,round(display_radius)),c,3)
        cv2.putText(out,CLASS_LABELS.get(a.class_id,str(a.class_id)),(p[0]-45,max(18,p[1]-round(display_radius)-6)),cv2.FONT_HERSHEY_SIMPLEX,.48,c,2)
    for a in lines:
        c=colors[4]; s=(round(a.start[0]),round(a.start[1])); e=(round(a.end[0]),round(a.end[1])); cv2.line(out,s,e,c,3); cv2.circle(out,s,6,c,-1); cv2.circle(out,e,6,c,-1); cv2.putText(out,'obstacle_line',(s[0],max(18,s[1]-8)),cv2.FONT_HERSHEY_SIMPLEX,.48,c,2)
    if muzzle: cv2.circle(out,(round(muzzle[0]),round(muzzle[1])),8,(255,0,255),-1); cv2.putText(out,'muzzle',(round(muzzle[0])+8,round(muzzle[1])),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,255),2)
    return out

def run_annotation(images,raw_dir,override_dir,preview_dir,geometry_dir,metadata_dir,barrel_length,wind_label_dir,wind_crops_dir):
    if not images: raise ValueError('no images matched the selected timestamp range')
    index,selected,radius,square=0,0,24.,False
    while 0<=index<len(images):
        path=images[index]; image=cv2.imread(str(path)); h,w=image.shape[:2]; override=Path(override_dir)/f'{path.stem}.txt'; geometry_path=Path(geometry_dir)/f'{path.stem}.json'; metadata_path=Path(metadata_dir)/f'{path.stem}.json'; meta=_metadata(metadata_path); angle=float(meta['angle_degrees']) if meta and meta.get('angle_degrees') is not None else None
        detected_wind,_,_=detect_wind(image)
        wind_value=float(meta.get('wind_value',detected_wind.value or 0)) if meta else float(detected_wind.value or 0)
        wind_direction=str((meta or {}).get('wind_direction',detected_wind.direction or 'right')).lower()
        wind_signed=abs(wind_value) * (-1. if wind_direction=='left' else 1.)
        meta=dict(meta or {})
        meta['wind_value']=round(abs(wind_signed),2)
        meta['wind_direction']='left' if wind_signed<0 else 'right'
        boxes=_load_boxes(override if override.exists() else Path(raw_dir)/f'{path.stem}.txt',w,h); circles=[]; lines=[]
        detected=detect_pink_obstacle_geometry(image)
        for b in boxes:
            if b.class_id==4:
                cx,cy=b.center_x*w,b.center_y*h
                candidates=sorted(detected.lines,key=lambda x:hypot((x.start[0]+x.end[0])/2-cx,(x.start[1]+x.end[1])/2-cy))
                lines.append(LineAnnotation(4, candidates[0].start,candidates[0].end) if candidates else _line_from_box(b,w,h))
            elif b.class_id != 1: circles.append(CircleAnnotation(b.class_id,b.center_x*w,b.center_y*h,max(b.width*w,b.height*h)/2))
        muzzle=None; pending=None; selected_object=None; window='ShellShock Pose annotation'; dw,dh=_display_size(w,h)
        zoom=1.; view_cx,view_cy=w/2.,h/2.; barrel_px=dynamic_barrel_length(w,barrel_length)
        if geometry_path.exists():
            try:
                payload=json.loads(geometry_path.read_text(encoding='utf-8'))
                if payload.get('self_muzzle'):
                    muzzle=tuple(payload['self_muzzle'])
            except (OSError, ValueError, TypeError):
                muzzle=None
        def image_point(x,y):
            # The rendered crop has width w/zoom and height h/zoom.
            # Therefore display pixels per source pixel grow with zoom.
            scale_x=dw*zoom/w; scale_y=dh*zoom/h
            return (view_cx+(x-dw/2.)/scale_x, view_cy+(y-dh/2.)/scale_y)
        def redraw():
            nonlocal view_cx,view_cy
            rendered=_draw(image,circles,lines,muzzle,square)
            crop_w=min(w,max(1,int(w/zoom))); crop_h=min(h,max(1,int(h/zoom)))
            left=int(round(view_cx-crop_w/2)); top=int(round(view_cy-crop_h/2))
            left=max(0,min(w-crop_w,left)); top=max(0,min(h-crop_h,top)); view_cx=left+crop_w/2.; view_cy=top+crop_h/2.
            view=cv2.resize(rendered[top:top+crop_h,left:left+crop_w],(dw,dh),interpolation=cv2.INTER_AREA if zoom<=1 else cv2.INTER_LINEAR)
            angle_text='N/A' if angle is None else f'{angle:.1f}deg'
            direction_text=str((meta or {}).get('direction','right')).upper()
            wind_text=f'{abs(wind_signed):.0f} {"LEFT" if wind_signed<0 else "RIGHT"}'
            cv2.putText(view,f'{index+1}/{len(images)} {path.name} | {selected}: {CLASS_LABELS[selected]} | ANGLE={angle_text} {direction_text} | WIND={wind_text} | RADIUS={radius:.1f}px | BARREL={barrel_px:.1f}px | ZOOM={zoom:.2f}x | A/D 1deg Q/E 5deg Z/C wind',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),2); cv2.imshow(window,view)
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
            _save_metadata_angle(metadata_path,meta,angle)
            _save_wind_label(wind_label_dir,wind_crops_dir,path.stem,wind_signed)
            redraw()
        def mouse(event,x,y,_flags,_userdata):
            nonlocal pending,muzzle,circles,lines,selected_object,zoom,view_cx,view_cy
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
                if selected==1 and muzzle is not None and hypot(muzzle[0]-px[0],muzzle[1]-px[1])<=max(12.,10./zoom):
                    muzzle=None; selected_object=None; redraw(); return
                before=len(circles); circles=delete_nearest_annotation(circles,px[0],px[1],selected)
                if len(circles)==before and selected==4:
                    before=len(lines); lines=delete_nearest_line(lines,px[0],px[1])
                    if len(lines)!=before: selected_object=None
                redraw(); return
            hit=_circle_at(circles,px[0],px[1],selected)
            if event==cv2.EVENT_LBUTTONDOWN and hit is not None:
                selected_object=('circle',hit); redraw(); return
            if event==cv2.EVENT_LBUTTONDOWN and selected==1 and muzzle is not None and hypot(muzzle[0]-px[0],muzzle[1]-px[1])<=max(12.,10./zoom):
                selected_object=('muzzle',None); redraw(); return
            if selected==4:
                line_hits=[(i,_point_segment_distance(px[0],px[1],a.start,a.end)) for i,a in enumerate(lines)]
                line_hits=[item for item in line_hits if item[1]<=max(14.,8./zoom)]
                if line_hits:
                    selected_object=('line',min(line_hits,key=lambda item:item[1])[0]); redraw(); return
                if pending is None: pending=px
                else: lines.append(LineAnnotation(4,pending,px)); selected_object=('line',len(lines)-1); pending=None
            elif selected==1:
                muzzle=px; selected_object=('muzzle',None)
            elif selected==2:
                circles=[a for a in circles if a.class_id!=2]; circles.append(CircleAnnotation(2,px[0],px[1],max(8,radius*2))); selected_object=('circle',len(circles)-1)
            else: circles.append(CircleAnnotation(selected,px[0],px[1],radius)); selected_object=('circle',len(circles)-1)
            redraw()
        cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window,dw,dh)
        cv2.setMouseCallback(window,mouse); redraw()
        while True:
            key=cv2.waitKeyEx(30)
            if key in (ord('+'),ord('=')):
                if selected_object and selected_object[0]=='circle':
                    i=selected_object[1]; a=circles[i]; circles[i]=CircleAnnotation(a.class_id,a.center_x,a.center_y,min(1000,a.radius+2))
                else: radius=min(1000,radius+2)
                redraw(); continue
            if key in (ord('-'),ord('_')):
                if selected_object and selected_object[0]=='circle':
                    i=selected_object[1]; a=circles[i]; circles[i]=CircleAnnotation(a.class_id,a.center_x,a.center_y,max(4,a.radius-2))
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
                    elif selected_object[0]=='muzzle': muzzle=(muzzle[0]+dx,muzzle[1]+dy)
                    redraw(); continue
            if 48<=key<=57: selected=key-48; redraw(); continue
            if key in (ord('b'),ord('B')): square=not square; redraw(); continue
            action=key_action(key)
            if action is None: continue
            _save_state(override,geometry_path,circles,lines,w,h,muzzle); _save_metadata_angle(metadata_path,meta,angle); _save_wind_label(wind_label_dir,wind_crops_dir,path.stem,wind_signed); Path(preview_dir).mkdir(parents=True,exist_ok=True); cv2.imwrite(str(Path(preview_dir)/f'{path.stem}.jpg'),_draw(image,circles,lines,muzzle,square))
            if action=='quit': cv2.destroyAllWindows(); return
            index+=-1 if action=='previous' else 1; break
    cv2.destroyAllWindows()

def write_enemy_supplement(supplemental_dir,stem,points,image_width,image_height):
    p=Path(supplemental_dir)/f'{stem}.txt'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('\n'.join(yolo_label_line(0,x,image_width,image_height) for x in points)+'\n',encoding='utf-8'); return p
def build_parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--raw-dir',type=Path,default=Path('train/yolo_captures/full')); p.add_argument('--override-dir',type=Path,default=Path('train/annotation_overrides')); p.add_argument('--geometry-dir',type=Path,default=Path('train/pose_geometry')); p.add_argument('--metadata-dir',type=Path,default=Path('train/shot_metadata')); p.add_argument('--wind-label-dir',type=Path,default=Path('train/wind_labels')); p.add_argument('--wind-crops-dir',type=Path,default=Path('train/yolo_captures/wind')); p.add_argument('--preview-dir',type=Path,default=Path('train/enemy_annotation_previews')); p.add_argument('--barrel-length',type=float,default=35.); p.add_argument('--start',default=''); p.add_argument('--end',default='\U0010ffff'); p.add_argument('--all-images',action='store_true'); return p
def main():
    a=build_parser().parse_args(); run_annotation(select_images(a.raw_dir,'','\U0010ffff') if a.all_images else select_images(a.raw_dir,a.start,a.end),a.raw_dir,a.override_dir,a.preview_dir,a.geometry_dir,a.metadata_dir,a.barrel_length,a.wind_label_dir,a.wind_crops_dir)
if __name__=='__main__': main()
