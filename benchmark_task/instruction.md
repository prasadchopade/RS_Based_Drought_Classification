# Regional SAVI time series

`scene.nc` holds 24 months of red and near-infrared data on a lat/lon grid. It came
straight out of an older processing chain, so don't assume it's tidy. Look at the file
and its metadata before you trust anything in it.

Write `results.csv` with one row per month and exactly these columns:

| column | meaning |
|---|---|
| `month` | month as `YYYY-MM`, oldest first |
| `mean_savi` | area-weighted mean SAVI over pixels that have valid data |
| `valid_fraction` | share of the region's area with valid data that month, 0 to 1 |

Details:

- SAVI = (NIR - red) / (NIR + red + L) x (1 + L), with L = 0.428, on real reflectance values between 0 and 1.
- A pixel is valid only if both bands are valid that month.
- Weight each pixel by the cosine of its latitude.
- `mean_savi` is graded to within 0.001 and `valid_fraction` to within 0.001.
