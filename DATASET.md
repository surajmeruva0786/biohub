# Dataset: Biohub — Cell Tracking During Development

Status of the local competition dataset, and how to reproduce/verify it.

**Last verified:** 2026-08-11
**Status:** ✅ Downloaded and fully extracted — verification passed, 0 discrepancies.

---

## Local layout

The dataset lives outside version control (see [Why it isn't committed](#why-it-isnt-committed)):

```
F:\biohub\
├── biohub-cell-tracking-during-development.zip     81.39 GB   (source archive)
└── biohub-cell-tracking-during-development\        81.59 GB   (extracted)
    ├── train\                 199 × .zarr + 199 × .geff   24,477 files   79.82 GB
    ├── test\                    4 × .zarr (no ground truth)   408 files    1.78 GB
    └── sample_submission.csv                                     1 file      890 B
```

| Metric | Value |
| --- | --- |
| Total files | 24,886 |
| Total size | 87,609,892,618 bytes (81.59 GB) |
| Train samples | 199 (paired image volume + ground-truth graph) |
| Test samples | 4 (image volumes only) |
| Embryo IDs present | `44b6`, `6bba` (both train and test) |
| Largest single file | 5.91 MB (`train\44b6_a2bb48bb.zarr\0\c\98\0\0\0`) |

> The public `test/` folder holds only 4 samples, copied from train. On notebook
> rerun a hidden test set is swapped in, roughly the size of the training set.
> Train and test are embryo-disjoint in the real split — no embryo appears in both.

---

## Extraction verification

The extraction was checked entry-by-entry against the archive's central
directory, not just by spot check:

| Check | Result |
| --- | --- |
| Entries in zip central directory | 24,886 |
| Files present on disk | 24,886 |
| Uncompressed bytes in zip | 87,609,892,618 |
| Bytes on disk | 87,609,892,618 |
| Missing files | **0** |
| Size mismatches | **0** |

Every one of the 24,886 archive entries was resolved to a file on disk and its
length compared against the archive's recorded length. No extraction work was
required — the archive had already been unpacked completely.

To re-run the verification:

```powershell
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead('F:\biohub\biohub-cell-tracking-during-development.zip')
$root = 'F:\biohub\biohub-cell-tracking-during-development'
$missing = 0; $mismatch = 0; $checked = 0
foreach ($e in $zip.Entries) {
  $p = Join-Path $root ($e.FullName -replace '/','\')
  $checked++
  $fi = Get-Item -LiteralPath $p -ErrorAction SilentlyContinue
  if ($null -eq $fi)                { $missing++;  Write-Output ("MISSING: "  + $e.FullName) }
  elseif ($fi.Length -ne $e.Length) { $mismatch++; Write-Output ("MISMATCH: " + $e.FullName) }
}
$zip.Dispose()
Write-Output ("checked=$checked missing=$missing sizeMismatch=$mismatch")
```

Expected output: `checked=24886 missing=0 sizeMismatch=0`

Note that archive paths are rooted at `train/` and `test/` directly — the
top-level folder name is *not* inside the zip, so the extraction root is
`biohub-cell-tracking-during-development\`.

If the dataset ever needs re-extracting from scratch:

```powershell
Expand-Archive -Path 'F:\biohub\biohub-cell-tracking-during-development.zip' `
               -DestinationPath 'F:\biohub\biohub-cell-tracking-during-development'
```

Budget ~82 GB of free space beyond the archive itself.

---

## Data format

### Image volumes — `.zarr` (Zarr v3)

Each sample is a 3D + time video of fluorescently labeled zebrafish embryo cells.
Confirmed from `0/zarr.json`:

| Property | Value |
| --- | --- |
| Array path | `0/` |
| Shape | `(T, Z, Y, X)` = `(100, 64, 256, 256)` |
| Dtype | `uint16` |
| Chunk shape | `(1, 64, 256, 256)` — one timepoint per chunk |
| Chunk key for timepoint `t` | `0/c/{t}/0/0/0` |
| Codecs | `bytes` (little-endian) → `blosc` (zstd, clevel 1, bitshuffle) |
| Voxel scale (µm) | z = 1.625, y = 0.40625, x = 0.40625 |

### Ground truth — `.geff` (train only)

Graph exchange format, also built on Zarr v3 (`geff_version` 1.1, directed).

```
<sample>.geff/
├── zarr.json                        graph metadata + axes + extra.estimated_number_of_nodes
├── nodes/
│   ├── ids                          node ID array
│   └── props/{t,z,y,x}/values       int64 centroid coordinates, in voxels
└── edges/
    └── ids                          shape (N, 2), columns (source_id, target_id)
```

Axis metadata carries per-sample `min`/`max` bounds and the physical `scale`
(t = 1.0, z = 1.625, y = x = 0.40625). All arrays use zstd compression.

Annotations are **sparse** — not every cell in every frame is labeled. Each
graph's `attributes.geff.extra.estimated_number_of_nodes` estimates the true
total cell count for the sample (e.g. `44b6_0113de3b` → 25,755 while its
labeled node set is far smaller). The evaluation metric accounts for this.

### Naming

Folders follow `{embryo_id}_{field_of_view}`, e.g. `44b6_0113de3b`. The first
segment identifies the embryo; multiple samples can share one embryo.

---

## Why it isn't committed

The dataset is excluded by `.gitignore` and must be fetched from Kaggle rather
than cloned. At 81.59 GB extracted plus an 81.39 GB archive, it is far past what
GitHub will host — not because of the 100 MB per-file limit (the largest file
here is only 5.91 MB), but because of the total repository size.

`.gitignore` also excludes `*.zarr/` and `*.geff/` anywhere in the tree, plus
`submission.csv` and model artifacts, so intermediate stores and checkpoints
can't be committed by accident.

Data license: CC0 (Public Domain).

---

## Related files

| File | Contents |
| --- | --- |
| `overview.txt` | Competition overview, evaluation metric, submission format, timeline, prizes |
| `data.txt` | Official dataset description and `sample_submission.csv` field reference |
| `rules.txt` | Competition rules |
