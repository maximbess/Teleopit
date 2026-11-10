"""First crossings and per-signal persistence, independently of controller aborts."""


class Events:
    def __init__(self, dt, persistence):
        self.dt, self.persistence = dt, persistence
        self.active = {}
        self.first = {}
        self.persistent = {}
        self.rows = []
        self.extrema = {}

    def update(self, time, phase, values, thresholds):
        for name, threshold in thresholds.items():
            value = values.get(name)
            if value is not None:
                self.extrema[name] = max(value, self.extrema.get(name, float("-inf")))
            if value is None or value < threshold:
                self.active.pop(name, None)
                continue
            if name not in self.active:
                row = {
                    "event": name,
                    "entity": name.split(":")[-1],
                    "threshold": threshold,
                    "observed_value": value,
                    "first_crossing_s": time,
                    "persistence_onset_s": None,
                    "phase": phase,
                    "resolution_s": self.dt,
                }
                self.active[name] = row
                self.rows.append(row)
                self.first.setdefault(name, row.copy())
            row = self.active[name]
            if (
                row["persistence_onset_s"] is None
                and time - row["first_crossing_s"] + self.dt >= self.persistence - 1e-9
            ):
                row["persistence_onset_s"] = time
                self.persistent.setdefault(name, row.copy())

    def reset_persistence(self):
        self.active.clear()


class Saturation:
    def __init__(self, names, dt, threshold):
        self.dt, self.threshold = dt, threshold
        self.hold_seconds = 0.0
        self.rows = {
            n: {
                "first_onset_s": None,
                "continuous_s": 0.0,
                "longest_s": 0.0,
                "hold_saturated_s": 0.0,
            }
            for n in names
        }

    def update(self, time, fractions, hold):
        self.hold_seconds += self.dt if hold else 0
        for name, value in fractions.items():
            row = self.rows[name]
            row["continuous_s"] = (
                row["continuous_s"] + self.dt if value >= self.threshold else 0.0
            )
            row["longest_s"] = max(row["longest_s"], row["continuous_s"])
            if value >= self.threshold:
                if row["first_onset_s"] is None:
                    row["first_onset_s"] = time
                row["hold_saturated_s"] += self.dt if hold else 0

    def summary(self):
        return {
            n: {
                **r,
                "fraction_of_hold": r["hold_saturated_s"] / self.hold_seconds
                if self.hold_seconds
                else None,
            }
            for n, r in self.rows.items()
        }
