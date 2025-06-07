<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->

<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License: MIT][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/arunoruto/reflectance-models"> <!-- TODO: Replace with actual link -->
    <!-- <img src="images/logo.png" alt="Logo" width="80" height="80"> TODO: Add logo if available -->
  </a>

  <h3 align="center">refmod</h3>

  <p align="center">
    A Python library for the Hapke photometric model, used for modeling light scattering from surfaces.
    <br />
    <a href="https://github.com/arunoruto/reflectance-models"><strong>Explore the docs »</strong></a> <!-- TODO: Replace with actual link -->
    <br />
    <br />
    <a href="https://github.com/arunoruto/reflectance-models">View Demo</a> <!-- TODO: Replace with actual link -->
    &middot;
    <a href="https://github.com/arunoruto/reflectance-models/issues/new?labels=bug&template=bug-report---.md">Report Bug</a> <!-- TODO: Replace with actual link -->
    &middot;
    <a href="https://github.com/arunoruto/reflectance-models/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a> <!-- TODO: Replace with actual link -->
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

<!-- TODO: Add screenshot if available -->
<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

`refmod` is a Python library dedicated to the Hapke photometric model, a widely used model in planetary science and remote sensing to describe the scattering of light from particulate surfaces.

The Hapke model, developed by Bruce Hapke, provides a theoretical framework to relate the observed reflectance of a surface to its physical properties, such as particle size, compaction, and single-scattering albedo. It accounts for various phenomena like opposition effect, shadow hiding, and multiple scattering.

`refmod` offers Python implementations of:

- The core Hapke reflectance equations.
- Various scattering functions (e.g., Legendre polynomials, Henyey-Greenstein).
- Functions to model the opposition effect.
- Utilities for common calculations and parameter conversions.

This library aims to provide researchers and students with an easy-to-use and well-documented tool for applying the Hapke model in their work.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

- [![Python][Python.org]][Python-url]
- [![NumPy][Numpy.org]][Numpy-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

To get `refmod` up and running on your local machine, follow these simple steps.

### Prerequisites

Ensure you have Python installed (version 3.6 or higher is recommended). You will also need pip to install packages.

### Installation

You can install `refmod` directly from PyPI:

```sh
pip install refmod
```

<!-- TODO: Update installation instructions if it's not on PyPI or has other dependencies -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

Here's a basic example of how to use `refmod` to calculate reflectance:

```python
import refmod
import numpy as np

# Define Hapke parameters
incidence_angle = 30  # degrees
emission_angle = 0    # degrees
phase_angle = 30      # degrees
ssa = 0.8             # single scattering albedo
# ... other parameters like Henyey-Greenstein asymmetry parameter, porosity, etc.

# Calculate reflectance
# reflectance = refmod.hapke_isotropic(incidence_angle, emission_angle, phase_angle, ssa) # Example function
# print(f"Reflectance: {reflectance}")
```

_For more detailed examples, please refer to the [Documentation](https://arunoruto.github.io/refmod/)_ <!-- TODO: Replace with actual documentation link -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->

## Roadmap

- [ ] Core Hapke model implementation.
- [ ] Implementation of common scattering phase functions.
- [ ] Functions for opposition effect modeling.
- [ ] `lumax`: Subpackage with JAX-based functions for GPU acceleration and automatic differentiation.
- [ ] `lumba`: Subpackage with Numba-jitted functions for performance enhancement on CPU.
- [ ] Comprehensive test suite.
- [ ] Detailed documentation and example notebooks.

See the [open issues](https://github.com/arunoruto/reflectance-models/issues) for a full list of proposed features (and known issues). <!-- TODO: Replace with actual link -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for details.

<!-- TODO: Create a LICENSE.txt file with the Unlicense text if it doesn't exist -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Project Link: [https://github.com/arunoruto/reflectance-models](https://github.com/arunoruto/reflectance-models) <!-- TODO: Replace with actual link -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- Hapke, B. (1981). Bidirectional reflectance spectroscopy: 1. Theory. Journal of Geophysical Research: Solid Earth, 86(B4), 3039-3054.
- Hapke, B. (1993). Theory of reflectance and emittance spectroscopy. Cambridge university press.
- Hapke, B. (2002). Bidirectional reflectance spectroscopy: 5. The coherent backscatter opposition effect and anisotropic scattering. Icarus, 157(2), 523-534.
- Hapke, B. (2012). Theory of reflectance and emittance spectroscopy (2nd ed.). Cambridge University Press.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- TODO: Update these links and add new ones as needed -->

[contributors-shield]: https://img.shields.io/github/contributors/arunoruto/reflectance-models.svg?style=for-the-badge
[contributors-url]: https://github.com/arunoruto/reflectance-models/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/arunoruto/reflectance-models.svg?style=for-the-badge
[forks-url]: https://github.com/arunoruto/reflectance-models/network/members
[stars-shield]: https://img.shields.io/github/stars/arunoruto/reflectance-models.svg?style=for-the-badge
[stars-url]: https://github.com/arunoruto/reflectance-models/stargazers
[issues-shield]: https://img.shields.io/github/issues/arunoruto/reflectance-models.svg?style=for-the-badge
[issues-url]: https://github.com/arunoruto/reflectance-models/issues
[license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/arunoruto/reflectance-models/blob/master/LICENSE
[Python.org]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[Numpy.org]: https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white
[Numpy-url]: https://numpy.org/

<!-- [product-screenshot]: images/screenshot.png -->
