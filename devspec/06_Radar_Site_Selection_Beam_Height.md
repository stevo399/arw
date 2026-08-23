
# Radar Site Selection and Beam Height Algorithm

Goal:
Select the radar that observes a storm at the lowest useful beam height.

## Beam Height Formula

Doviak and Zrnic equation 2.28b, under the 4/3 effective earth radius model:

beam_height = sqrt(r^2 + R^2 + 2*r*R*sin(elevation)) - R + radar_elevation

where:
  r = slant range from the radar
  R = effective earth radius = 4/3 * 6371 km = 8494.7 km
  elevation = beam elevation angle, lowest tilt ~= 0.5 degrees

The effective radius accounts for standard atmospheric refraction bending the
beam downward relative to a straight line. Using the true earth radius of
6371 km overestimates beam height by up to 1766 m at 300 km and wrongly
rejects radar sites beyond 306 km.

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
