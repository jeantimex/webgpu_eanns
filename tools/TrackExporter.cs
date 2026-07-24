/// TrackExporter — dumps a track scene of this Unity project to the JSON format
/// used by the WebGPU port (webgpu_eanns, public/tracks/*.json):
///   { "name": ..., "start": { "x", "y", "angleDeg" },
///     "checkpoints": [[x, y], ...], "walls": [[x1, y1, x2, y2], ...] }
///
/// Usage:
///   1. Copy this file to UnityProject/Assets/Editor/TrackExporter.cs
///   2. Open a track scene (Assets/Scenes/Tracks/Track1..4)
///   3. Menu "Tools/EANNs/Export Open Track Scene" (or "Export All Track Scenes")
///   4. JSON is written to <project>/ExportedTracks/<SceneName>.json
///
/// What gets exported:
///   - checkpoints: TrackManager's Checkpoint children, in hierarchy order
///     (checkpoints[0] is the start line, as TrackManager expects).
///   - start: PrototypeCar position and eulerAngles.z (our angleDeg is the same
///     CCW-positive convention with 0 = +Y).
///   - walls: every non-trigger Collider2D in the scene (cars excluded), as
///     segments: BoxCollider2D -> its 4 edges, Edge/PolygonCollider2D -> their
///     point polylines. Curved walls therefore export exactly if they are
///     Edge/Polygon colliders; box-approximated curves export as their boxes.
using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class TrackExporter
{
    [MenuItem("Tools/EANNs/Export Open Track Scene")]
    private static void ExportOpenScene()
    {
        string json = BuildTrackJson(UnityEngine.SceneManagement.SceneManager.GetActiveScene().name);
        WriteJson(UnityEngine.SceneManagement.SceneManager.GetActiveScene().name, json);
    }

    [MenuItem("Tools/EANNs/Export All Track Scenes")]
    private static void ExportAllScenes()
    {
        string originalPath = EditorSceneManager.GetActiveScene().path;
        string scenesDir = Application.dataPath + "/Scenes/Tracks";
        foreach (string file in Directory.GetFiles(scenesDir, "*.unity"))
        {
            string scenePath = "Assets/Scenes/Tracks/" + Path.GetFileName(file);
            EditorSceneManager.OpenScene(scenePath);
            string sceneName = Path.GetFileNameWithoutExtension(file);
            WriteJson(sceneName, BuildTrackJson(sceneName));
        }
        EditorSceneManager.OpenScene(originalPath);
    }

    private static void WriteJson(string sceneName, string json)
    {
        string dir = Application.dataPath + "/../ExportedTracks";
        Directory.CreateDirectory(dir);
        string path = dir + "/" + sceneName + ".json";
        File.WriteAllText(path, json);
        Debug.Log("Track exported to " + path);
    }

    private static string BuildTrackJson(string trackName)
    {
        TrackManager tm = Object.FindObjectOfType<TrackManager>();
        if (tm == null) throw new System.InvalidOperationException("No TrackManager in scene " + trackName);
        if (tm.PrototypeCar == null) throw new System.InvalidOperationException("TrackManager.PrototypeCar not set in " + trackName);

        StringBuilder sb = new StringBuilder();
        sb.Append("{\n");
        sb.Append("  \"name\": \"").Append(trackName).Append("\",\n");

        Transform start = tm.PrototypeCar.transform;
        sb.Append("  \"start\": { \"x\": ").Append(F(start.position.x))
          .Append(", \"y\": ").Append(F(start.position.y))
          .Append(", \"angleDeg\": ").Append(F(start.eulerAngles.z)).Append(" },\n");

        sb.Append("  \"checkpoints\": [");
        Checkpoint[] checkpoints = tm.GetComponentsInChildren<Checkpoint>();
        for (int i = 0; i < checkpoints.Length; i++)
        {
            Vector3 p = checkpoints[i].transform.position;
            sb.Append(i == 0 ? "\n    [" : ",\n    [").Append(F(p.x)).Append(", ").Append(F(p.y)).Append("]");
        }
        sb.Append("\n  ],\n");

        sb.Append("  \"walls\": [");
        bool first = true;
        foreach (Collider2D col in Object.FindObjectsOfType<Collider2D>())
        {
            if (col.isTrigger) continue;
            if (col.GetComponentInParent<CarMovement>() != null) continue; // the prototype car
            BoxCollider2D box = col as BoxCollider2D;
            if (box != null)
            {
                // Emit the box's 4 edges: the collider surface is the actual wall.
                Vector2 hs = box.size * 0.5f;
                Vector2 o = box.offset;
                Vector2[] corners = new Vector2[]
                {
                    o + new Vector2(-hs.x, -hs.y), o + new Vector2(hs.x, -hs.y),
                    o + new Vector2(hs.x, hs.y), o + new Vector2(-hs.x, hs.y),
                };
                for (int i = 0; i < 4; i++)
                    AppendSegment(sb, ref first, box.transform, corners[i], corners[(i + 1) % 4]);
                continue;
            }
            EdgeCollider2D edge = col as EdgeCollider2D;
            if (edge != null)
            {
                for (int i = 0; i + 1 < edge.points.Length; i++)
                    AppendSegment(sb, ref first, edge.transform, edge.points[i], edge.points[i + 1]);
                continue;
            }
            PolygonCollider2D poly = col as PolygonCollider2D;
            if (poly != null)
            {
                for (int path = 0; path < poly.pathCount; path++)
                {
                    Vector2[] points = poly.GetPath(path);
                    for (int i = 0; i < points.Length; i++)
                        AppendSegment(sb, ref first, poly.transform, points[i], points[(i + 1) % points.Length]);
                }
                continue;
            }
            Debug.LogWarning("TrackExporter: skipping unsupported collider on " + col.name);
        }
        sb.Append("\n  ]\n}\n");
        return sb.ToString();
    }

    private static void AppendSegment(StringBuilder sb, ref bool first, Transform t, Vector2 localA, Vector2 localB)
    {
        Vector2 a = t.TransformPoint(localA);
        Vector2 b = t.TransformPoint(localB);
        sb.Append(first ? "\n    [" : ",\n    [");
        first = false;
        sb.Append(F(a.x)).Append(", ").Append(F(a.y)).Append(", ").Append(F(b.x)).Append(", ").Append(F(b.y)).Append("]");
    }

    private static string F(float v)
    {
        return v.ToString("R", CultureInfo.InvariantCulture);
    }
}
