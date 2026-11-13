# Data of the interactive figures

Written by the tools of [clipasso3d](https://github.com/tarhanefe/clipasso3d) (branch `mainv2`), run from its root:

```bash
SITE=../CLIPasso3DWebsite/static
# lines x abstraction grid + gallery (all objects), connectivity grid, index, metrics
python tools/export_web.py runs --out $SITE/data --family ablation --pattern "abl_L{col}_{row}"
python tools/export_web.py runs --out $SITE/data --family connect --pattern "conn_L{col}_{row}" \
    --row_pattern "off=abl_L{col}_normal" --rows off weld loss01 loss1 --cols 5 20 50 200
python tools/export_web.py index --out $SITE/data --banner_object boat \
    --connect_rows "off:no constraint" "weld:weld only" "loss01:loss 0.1 + weld" "loss1:loss 1.0 + weld" --connect_cols 5 20 50 200
python tools/export_web.py metrics --out $SITE/data
# ground-truth meshes (Blender); the banner under the title is drawn live from these files
for o in $(ls meshes/objaverse/*.glb); do blender -b -P tools/export_gt_glb.py -- --mesh $o --out $SITE/data/gt/$(basename $o); done
```

Layout: `index.json`, `metrics.json`, `gt/<object>.glb`, and `<family>/<object>/{runs.json, finals.bin, history.bin}`
(int16 control points, see `tools/export_web.py`). Missing objects or families show "results coming soon".
