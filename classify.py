from pathlib import Path
import shutil
from PIL import Image
# 原始截图所在文件夹
src_dir = Path(r"C:\Users\jingh\Desktop\game\shellshock\output")
obj_dir = Path(r"C:\Users\jingh\Desktop\game\shellshock\train")
# 输出文件夹
raw_dir = obj_dir / "raw"
annotated_dir = obj_dir / "annotated"
other_dir = obj_dir / "other"
raw_cropped_dir = obj_dir / "raw_cropped"

for folder in [
    raw_dir,
    raw_cropped_dir,
    annotated_dir,
    other_dir
]:
    folder.mkdir(exist_ok=True)


# =========================
# 裁剪参数
# 原图：2048 × 1152
# 保留：y = 0 ~ 988
# =========================
CROP_BOTTOM = 1800


for file in src_dir.iterdir():

    if not file.is_file():
        continue

    # 只处理常见图片
    if file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp"]:
        continue

    name = file.stem.lower()

    # =========================
    # RAW
    # =========================
    if name.endswith("_raw"):

        # 先复制原图
        raw_target = raw_dir / file.name
        shutil.copy2(file, raw_target)

        # 再生成裁剪版本
        try:
            with Image.open(file) as img:

                width, height = img.size

                # 防止某些图片高度不足
                crop_bottom = min(CROP_BOTTOM, height)

                cropped = img.crop(
                    (
                        0,              # left
                        0,              # top
                        width,          # right
                        crop_bottom     # bottom
                    )
                )

                cropped_target = raw_cropped_dir / file.name
                cropped.save(cropped_target)

                print(
                    f"[RAW] {file.name} "
                    f"{width}x{height} -> "
                    f"{cropped.width}x{cropped.height}"
                )

        except Exception as e:
            print(f"[ERROR] 裁剪失败: {file.name}: {e}")


    # =========================
    # ANNOTATED
    # =========================
    elif name.endswith("_annotated"):

        target = annotated_dir / file.name
        shutil.copy2(file, target)

        print(f"[ANNOTATED] {file.name}")


    # =========================
    # OTHER
    # =========================
    else:

        target = other_dir / file.name
        shutil.copy2(file, target)

        print(f"[OTHER] {file.name}")


print("\n处理完成。")