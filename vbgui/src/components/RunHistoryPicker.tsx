// V7-K7 — run-history picker for the warm-start path. When the user
// flips the warm-start checkbox in TopBar, this picker decides WHICH
// past run_id to continue from. "(latest)" preserves the original
// behaviour (use lastTrainRunId).

import { HelpIcon } from "@/components/HelpIcon";
import { T } from "@/theme";

export interface RunHistoryPickerProps {
  history: readonly string[];
  selected: string | null;
  onSelect: (runId: string | null) => void;
}

export function RunHistoryPicker({
  history, selected, onSelect,
}: RunHistoryPickerProps): JSX.Element {
  return (
    <div data-testid="run-history-picker"
         style={{ display: "flex", alignItems: "center", gap: 6,
                  fontSize: 12, padding: "4px 8px",
                  background: T.surface,
                  borderTop: `1px solid ${T.border}`,
                  color: T.text,
                  fontFamily: T.font }}>
      <strong style={{ color: T.accent }}>warm-start from</strong>
      <HelpIcon topic="warm_start_history" />
      <select data-testid="run-history-select"
              value={selected ?? ""}
              onChange={(e) => onSelect(
                e.target.value === "" ? null : e.target.value)}
              style={{
                minWidth: 240,
                color: T.text,
                background: T.surface3,
                border: `1px solid ${T.border}`,
              }}>
        <option value="">(latest)</option>
        {history.map((id) => (
          <option key={id} value={id}>{id}</option>
        ))}
      </select>
      <span data-testid="run-history-count"
            style={{ color: T.textSecondary, fontSize: 11, fontWeight: "bold" }}>
        {history.length} run{history.length === 1 ? "" : "s"} in history
      </span>
    </div>
  );
}
