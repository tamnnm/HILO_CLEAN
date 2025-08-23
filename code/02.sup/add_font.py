import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

# Your font path goes here
font_dirs = '/work/users/tamnnm/code/2.sup/Hershey_font_TTF/ttf/'
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

fp = matplotlib.font_manager.FontProperties(
    fname=font_dirs)
print(fp.get_name())

matplotlib.font_manager._load_fontmanager(try_read_cache=False)


def _load_fontmanager(*, try_read_cache=True):
    fm_path = Path(
        mpl.get_cachedir(), f"fontlist-v{FontManager.__version__}.json")
    if try_read_cache:
        try:
            fm = json_load(fm_path)
        except Exception as exc:
            pass
        else:
            if getattr(fm, "_version", object()) == FontManager.__version__:
                _log.debug("Using fontManager instance from %s", fm_path)
                return fm
    fm = FontManager()
    json_dump(fm, fm_path)
    _log.info("generated new fontManager")
    return fm
