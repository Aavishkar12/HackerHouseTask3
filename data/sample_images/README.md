# data/sample_images — photo layout

Drop this whole `sample_images` folder into `data/` in the project, so
you end up with `data/sample_images/me.jpg`, `data/sample_images/refs/`,
and so on.

These files are **git-ignored on purpose** — they are real photos of real
people and the repo is public, so they never get committed and never
appear in a fresh clone. Keep this zip somewhere safe; it is the only
copy of the dataset outside your machine.

## Layout

```
data/sample_images/
├── me.jpg                  aviiii — single test photo
├── salman.jpg              public figure — single test photo
├── two_people.jpg          two faces in one frame (impostor test)
├── refs/                   GALLERY A — aviiii, 6 photos
│   ├── ref00_classroom.jpg
│   ├── ref01_mirror.jpg
│   ├── ref02_bed.jpg
│   ├── ref03_outdoor.jpg
│   ├── ref04_indoor.jpg
│   └── ref05_indoor2.jpg
└── public_figure/          GALLERY B — Salman Khan, 7 photos
    ├── ref00_blazer.jpg
    ├── ref01_navy_tee.jpg
    ├── ref02_black_henley.jpg
    ├── ref03_blue_check.jpg
    ├── ref04_being_human.jpg
    ├── ref05_bw_young.jpg
    └── ref06_waving.jpg
```

**`refs/` and `public_figure/` must never be mixed.** A gallery
represents exactly one identity; combining two people produces a gallery
that matches both, which silently destroys identification.

`me.jpg` is a copy of `refs/ref00_classroom.jpg` and `salman.jpg` is a
copy of `public_figure/ref00_blazer.jpg`, so matching either against its
own gallery gives distance 0.000. That is expected, not a bug — they are
convenience single-image inputs for `demo_encode.py`.

## Build both galleries

```bash
# Gallery A — you
python scripts/build_gallery.py

# Gallery B — the public figure
python scripts/build_gallery.py --refs-dir data/sample_images/public_figure \
                                --out data/output/gallery_public_figure.json
```

## Measured on these exact photos

Built and cross-checked before this zip was made (ArcFace, cosine
distance, tolerance 0.65):

| Check | Distance | Result |
|---|---|---|
| worst leave-one-out, `refs/` | 0.538 | within tolerance (headroom 0.112) |
| worst leave-one-out, `public_figure/` | 0.468 | within tolerance (headroom 0.182) |
| `me.jpg` vs public figure gallery | 0.833 | correctly NO MATCH |
| `salman.jpg` vs your gallery | 0.887 | correctly NO MATCH |
| `two_people.jpg` face 1 (aviiii) vs your gallery | 0.346 | MATCH |
| `two_people.jpg` face 2 (friend) vs your gallery | 0.758 | correctly NO MATCH |

Worst true match 0.538, closest impostor 0.758 — a margin of 0.220 with
the threshold at 0.65 sitting between them. Every photo in both folders
was confirmed to produce a usable face encoding.

The `ref05_bw_young.jpg` photo is decades older than the rest and is the
weakest reference at 0.468; it still lands comfortably inside tolerance
and widens the gallery's coverage, so it is worth keeping.

## The honest no-match demo

`two_people.jpg` face 2 and `me.jpg` vs the public-figure gallery are
both genuine rejections. Recording one of those is worth more than
another success — it shows the system reports "not this person" rather
than forcing a match.
