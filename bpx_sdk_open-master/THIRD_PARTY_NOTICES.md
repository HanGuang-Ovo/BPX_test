# Third-Party Notices

The BPX SDK Open source tree does not intentionally vendor third-party source
code. The precompiled native libraries and Python wheels may dynamically link
to operating-system, C/C++, and Python runtime components. Those components
remain subject to their respective licenses and are not relicensed under the
project's Apache License 2.0.

Before publishing a release, maintainers must audit the native implementation
and all generated artifacts for statically linked or bundled third-party code,
update this file with every applicable copyright and license notice, and
generate the release SBOM. This file must be distributed with the project
LICENSE and NOTICE files.
