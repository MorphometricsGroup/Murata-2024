import Metashape
import csv
chunk = Metashape.app.document.chunk
pc = chunk.point_cloud
pts = pc.points
tracks = pc.tracks
coords = []
colors = []
for i in range(len(pts)):
    pt_coord = pts[i].coord
    track_id = pts[i].track_id
    color = tracks[track_id].color
    coords.append(pt_coord)
    colors.append(color)

with open('coord.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(coords)

with open('colors.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(coords)
