# Data of the interactive figures

Written by the tools of [clipasso3d](https://github.com/tarhanefe/clipasso3d) (branch `mainv2`), run from its root:

```bash
SITE=../CLIPasso3DWebsite/static
# lines x abstraction grid + gallery (all objects), connectivity grid, index, metrics
python tools/export_web.py runs --out $SITE/data --family ablation --pattern "abl_L{col}_{row}"
python tools/export_web.py runs --out $SITE/data --family connect --pattern "<connect run pattern>" \
    --rows off loss loss_weld --cols 5 20 50 200
python tools/export_web.py index --out $SITE/data --banner_object boat \
    --connect_rows "off:no constraint" "loss:connectivity loss" "loss_weld:loss + weld" --connect_cols 5 20 50 200
python tools/export_web.py metrics --out $SITE/data
# ground-truth meshes (Blender) and the banner under the title
for o in $(ls meshes/objaverse/*.glb); do blender -b -P tools/export_gt_glb.py -- --mesh $o --out $SITE/data/gt/$(basename $o); done
python tools/export_banner.py --object boat --turntable turntables/boat --frame 9 \
    --runs runs/abl_L5_normal runs/abl_L20_normal runs/abl_L50_normal runs/abl_L200_normal --out $SITE/banner
```

Layout: `index.json`, `metrics.json`, `gt/<object>.glb`, and `<family>/<object>/{runs.json, finals.bin, history.bin}`
(int16 control points, see `tools/export_web.py`). Missing objects or families show "results coming soon".
