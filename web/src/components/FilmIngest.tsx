import { useState } from "react";
import { api } from "../api";
import type { Play } from "../types";

interface Props {
  onAnalyzed: (play: Play, defenseTeam: string) => void;
  onError: (message: string | null) => void;
}

export function FilmIngest({ onAnalyzed, onError }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [defenseTeam, setDefenseTeam] = useState("");
  const [offenseTeam, setOffenseTeam] = useState("");
  const [down, setDown] = useState("");
  const [distance, setDistance] = useState("");
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);

  const pickFile = (next: File | null) => {
    setFile(next);
    onError(null);
  };

  const submit = async () => {
    if (!file) {
      onError("Drop an MP4 clip or a still before analyzing.");
      return;
    }
    if (!defenseTeam.trim()) {
      onError("Type the defense team so the play is filed under that name.");
      return;
    }
    onError(null);
    setBusy(true);
    try {
      const result = await api.ingest({
        file,
        defense_team: defenseTeam,
        offense_team: offenseTeam,
        down,
        distance,
      });
      const play = result.plays[0];
      if (!play) {
        onError("The clip was accepted but no play came back.");
        return;
      }
      onAnalyzed(play, defenseTeam.trim());
    } catch (error) {
      onError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="card ingest-card">
      <div className="ingest-copy">
        <p className="eyebrow">Film room</p>
        <h2>Drop a clip. Get the tracking.</h2>
        <p className="muted">
          Type the team, then drop film. Player boxes and the field minimap follow
          every frame of the clip.
        </p>
      </div>

      <div className="ingest-form">
      <label
        className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          pickFile(event.dataTransfer.files[0] ?? null);
        }}
      >
        <input
          type="file"
          accept="video/mp4,video/quicktime,image/jpeg,image/png,image/webp"
          onChange={(event) => pickFile(event.target.files?.[0] ?? null)}
        />
        <strong>{file ? file.name : "Drop film here"}</strong>
        <span>MP4, MOV, or a still from All-22</span>
      </label>

      <div className="ingest-grid">
        <label className="field-label">
          Defense team
          <input
            type="text"
            value={defenseTeam}
            onChange={(event) => setDefenseTeam(event.target.value)}
            placeholder="Type the defense team"
            autoComplete="off"
            required
          />
        </label>
        <label className="field-label">
          Offense team
          <input
            type="text"
            value={offenseTeam}
            onChange={(event) => setOffenseTeam(event.target.value)}
            placeholder="Type the offense team"
            autoComplete="off"
          />
        </label>
        <label className="field-label">
          Down
          <input
            type="number"
            min={1}
            max={4}
            value={down}
            onChange={(event) => setDown(event.target.value)}
            placeholder="3"
          />
        </label>
        <label className="field-label">
          Distance
          <input
            type="number"
            min={1}
            value={distance}
            onChange={(event) => setDistance(event.target.value)}
            placeholder="7"
          />
        </label>
      </div>

      <button className="play-button ingest-submit" type="button" disabled={busy} onClick={() => void submit()}>
        {busy ? "Tracking every frame…" : "Analyze play"}
      </button>
      {busy && (
        <p className="muted ingest-wait">
          Player tracking runs on each frame of the clip, then the minimap is
          overlaid on the film.
        </p>
      )}
      </div>
    </section>
  );
}
