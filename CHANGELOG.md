# CHANGELOG


## v0.2.0 (2025-09-09)

### Features

- Implement inversion in the Hapke class
  ([`8160481`](https://github.com/arunoruto/reflectance-models/commit/81604817a276b5eb3b1bba1e190aa8f55992bb0a))


## v0.1.0 (2025-09-09)

### Bug Fixes

- Move inverse to hapke and create iterations over each pixel
  ([`cf20a75`](https://github.com/arunoruto/reflectance-models/commit/cf20a755f8f4094333b1d270079450219b4c1956))

### Chores

- Cleanup refmod
  ([`4cff71e`](https://github.com/arunoruto/reflectance-models/commit/4cff71eb7c5b8f8a560aff42071316a46f354107))

Change implementation to be more modular and be similar to the one of lumax.

- Update project config files
  ([`658c523`](https://github.com/arunoruto/reflectance-models/commit/658c523c7c1ce61c14261190382e2efa2570a61b))

### Continuous Integration

- Modify sphinx
  ([`d040a4d`](https://github.com/arunoruto/reflectance-models/commit/d040a4dd277a4cbaf4bf76d30aa178d228ab25a9))

- Update group name of the actions
  ([`3a92d06`](https://github.com/arunoruto/reflectance-models/commit/3a92d067bde13a39f10d968814d2ac991dfaf2f4))

- Update lock file
  ([`73d3ad3`](https://github.com/arunoruto/reflectance-models/commit/73d3ad3e9d82a55e09885a2f29cf1f3b15b2501c))

- Update lock file
  ([`12f41c0`](https://github.com/arunoruto/reflectance-models/commit/12f41c05c705eb0947090c1c77c83ee561bebc47))

- Update lock file
  ([`29fa19e`](https://github.com/arunoruto/reflectance-models/commit/29fa19e70be037137f024c44adb82b72d8d55438))

- Update lock file
  ([`0739675`](https://github.com/arunoruto/reflectance-models/commit/073967551badafc0a7dae1c1501fa7ba0c7a0366))

- Update lock file
  ([`ef2d573`](https://github.com/arunoruto/reflectance-models/commit/ef2d57361251e35eb1b70be051a68449b8ba8931))

- Update lock file
  ([`f737946`](https://github.com/arunoruto/reflectance-models/commit/f737946a73bbd1f01c87f5f29f1a2e7538aa61dd))

- Update lock file
  ([`4cdf42c`](https://github.com/arunoruto/reflectance-models/commit/4cdf42cc71aefa3dadd16fbf1ea7f2e1e65e92ea))

- Update lock file
  ([`fc02ba0`](https://github.com/arunoruto/reflectance-models/commit/fc02ba07acb8b7fe01e1434d7ee5ab1487e2907a))

- Update lock file
  ([`768889a`](https://github.com/arunoruto/reflectance-models/commit/768889a5226bc94c4e041957962ffe251c3a36e2))

- Update lock file
  ([`24f746f`](https://github.com/arunoruto/reflectance-models/commit/24f746f929d4d1918fca1ca44dd4b2fe909480c4))

- Update lock file
  ([`40e9488`](https://github.com/arunoruto/reflectance-models/commit/40e9488bd3b92fdb11791eb9b21a34da3c543928))

- Update lock file
  ([`0c5eccb`](https://github.com/arunoruto/reflectance-models/commit/0c5eccb044ba89bfc433194adc055c1f0d6f2274))

- Update lock file
  ([`f2d1a00`](https://github.com/arunoruto/reflectance-models/commit/f2d1a00f4837d89b1fa03bf91a3b5f4641de75c3))

- Update lock file
  ([`934c545`](https://github.com/arunoruto/reflectance-models/commit/934c5459d7c0324ab307137182ec66cbec0f21d9))

### Documentation

- Fix readme
  ([`a6f616b`](https://github.com/arunoruto/reflectance-models/commit/a6f616b58f011bad4a466dbc7992cb32b0425845))

- Refactor docs
  ([`4fec6b4`](https://github.com/arunoruto/reflectance-models/commit/4fec6b4d0a10fef37d78e4826f295bd35121b1b0))

- Reformat docs
  ([`7f38328`](https://github.com/arunoruto/reflectance-models/commit/7f38328c068872cf4d761bed8d38e2e97d5dda44))

- Update README with correct repo name
  ([`57a857d`](https://github.com/arunoruto/reflectance-models/commit/57a857d33bb3bf3f19bef05fa2d51d37e565465f))

### Features

- Add lumba optional dependency
  ([`9ba1cff`](https://github.com/arunoruto/reflectance-models/commit/9ba1cffda5da47c2b2ddcf03f91dc4256da7bd4b))

- Introduce pydantic for the Hapke class
  ([`3612105`](https://github.com/arunoruto/reflectance-models/commit/361210598822ef514110618c5b254d0966d088d0))

- Parallelize the project for each pixel and wavelength
  ([`9e64d4a`](https://github.com/arunoruto/reflectance-models/commit/9e64d4a5a2a9ac614b39c4190c0fa11edd28355c))

Moved the logic to look at each data point (w,h,lambda) as a parameter. This means we can process
  the whole image stack as a single vector to be optimized against. One problem: we need to repeat
  the incidence, emission, and normal vectors for each wavelength entry. If the image has n data
  points (n=w*h*lambda), then the vectors should have a resulting length of 3xn. The vector shape
  has also been reversed, i.e., the spatial dimension has been moved back so numpy's broadcasting
  can be utilized more!

- Refactor jit to vectorized functions and use levenberg pixelwise
  ([`ced1bb9`](https://github.com/arunoruto/reflectance-models/commit/ced1bb9335300b26315421366c90b7d53708a112))

- Unify models into one and refactor the project + testing
  ([`f521a23`](https://github.com/arunoruto/reflectance-models/commit/f521a23571011c361758bf899d453ab7d716f30f))

### Testing

- Add pytest cov
  ([`589fd34`](https://github.com/arunoruto/reflectance-models/commit/589fd344c7ecbe7b9a60942c32c3d8259dd2cb90))
