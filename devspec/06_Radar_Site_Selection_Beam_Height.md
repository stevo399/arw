
# Radar Site Selection and Beam Height Algorithm

Goal:
Select the radar that observes a storm at the lowest useful beam height.

## Beam Height Formula

beam_height =
distance * tan(elevation_angle) +
(distance² / (2 * earth_radius)) +
radar_elevation

Typical values:
lowest elevation angle ≈ 0.5°
earth radius ≈ 6371 km

## Algorithm

For each radar site:

1. compute distance to storm target
2. compute beam height at target
3. reject if beam height too high (>10 km)
4. rank remaining radars by lowest beam height

Best radar = lowest beam height over storm.

## Pseudocode

for radar in radars:
    distance = calc_distance(radar, target)
    beam_height = calc_beam_height(distance)
    if beam_height < threshold:
        candidates.append(radar)

best_radar = min(candidates, key=beam_height)
