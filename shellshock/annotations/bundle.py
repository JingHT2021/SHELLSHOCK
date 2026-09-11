"""Resolve labels and metadata beside their image before historical fallbacks."""
from pathlib import Path
from shellshock.config.paths import DATA_ROOT


def annotation_paths(image_path, root=DATA_ROOT):
    image=Path(image_path);root=Path(root)
    bundle=image.parent.parent if image.parent.name in {'images','full'} else root/'annotate'
    candidates=[bundle,root/'annotate',root/'yolo_captures']
    for base in candidates:
        paths=tuple(base/folder/f'{image.stem}{suffix}' for folder,suffix in
                    (('labels','.txt'),('pose_geometry','.json'),('metadata','.json')))
        if any(p.exists() for p in paths):
            return paths
    return tuple(bundle/folder/f'{image.stem}{suffix}' for folder,suffix in
                 (('labels','.txt'),('pose_geometry','.json'),('metadata','.json')))
