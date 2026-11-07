# Third-party robot model

Model: NEURA LARA 5, distributed through the RoboDK robot library.
Source page: https://robodk.com/robot/Neura/LARA-5
Source download: https://cdn.robodk.com/downloads-library/library-robots/Neura-LARA-5.robot
Retrieved for this project: 2026-10-26.
SHA-256: 4dc8df5ff0e8396c86a2388f9f0a582ba3d8c34c9a310fb8f98d290a25dee3c3

Setup downloads the unmodified .robot source and verifies its checksum. The OBJ
geometries under assets/robots/lara5/assets are derived from its tessellation, with
units converted and triangle winding aligned to the packed source normals.
The MJCF joint chain uses the kinematic table and joint ranges decoded from
that file. Other model assumptions are documented in parameters.json.

No additional license grant for these third-party model data is made by the
Stack Lab MIT license. No manufacturer endorsement, certification or validation
is implied. This package is not an official NEURA or RoboDK simulation release.
