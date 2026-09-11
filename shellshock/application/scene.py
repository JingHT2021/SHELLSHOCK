"""Shared interpretation of live frames, saved frames, and manual edits."""
from dataclasses import dataclass
from shellshock.annotations.conversion import annotations_to_world_with_diagnostics, yolo_detections_to_annotations
from shellshock.perception.wind import detect_wind


@dataclass
class AnalyzedFrame:
    scene: object
    world: object
    muzzle: object
    diagnostics: list
    wind_value: float
    wind_direction: str


def analyze_frame(image, *, detections=(), scene=None, wind_override=None):
    if scene is None:
        scene=yolo_detections_to_annotations(detections,image.shape[1],image.shape[0])
    world,muzzle,diagnostics=annotations_to_world_with_diagnostics(scene,image)
    metadata=scene.metadata
    if wind_override is not None:
        signed=float(wind_override)
        value=abs(signed)
        direction='left' if signed<0 else 'right'
    elif metadata.get('wind_value') is not None and metadata.get('wind_direction') in {'left','right'}:
        value=abs(float(metadata['wind_value']));direction=metadata['wind_direction']
    else:
        wind,_,_=detect_wind(image)
        value=float(wind.value or 0);direction=wind.direction or 'right'
        metadata['wind_source']='image' if wind.value is not None else 'assumed_zero'
    metadata.update(wind_value=value,wind_direction=direction)
    return AnalyzedFrame(scene,world,muzzle,diagnostics,value,direction)
