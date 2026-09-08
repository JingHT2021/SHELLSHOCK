"""Build a manifest aligning wind crops with manually reviewed wind labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def prepare(label_dir: Path, output_dir: Path) -> int:
    rows=[]
    for label_path in sorted(Path(label_dir).glob('*.json')):
        payload=json.loads(label_path.read_text(encoding='utf-8'))
        image=Path(payload['image'])
        if not image.exists():
            continue
        rows.append({'stem':payload.get('stem',label_path.stem), 'image':str(image), 'wind_value':float(payload['wind_value']), 'wind_direction':payload['wind_direction'], 'wind_signed':float(payload['wind_signed'])})
    output_dir.mkdir(parents=True,exist_ok=True)
    manifest=output_dir/'manifest.csv'
    with manifest.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=('stem','image','wind_value','wind_direction','wind_signed'))
        writer.writeheader(); writer.writerows(rows)
    (output_dir/'dataset.json').write_text(json.dumps({'count':len(rows),'target':'wind_signed','manifest':str(manifest)},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Wind RNN manifest: {len(rows)} samples -> {manifest}')
    return len(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label-dir',type=Path,default=Path('train/wind_labels'))
    parser.add_argument('--output-dir',type=Path,default=Path('train/wind_rnn_dataset'))
    args=parser.parse_args(); prepare(args.label_dir,args.output_dir)


if __name__=='__main__': main()
