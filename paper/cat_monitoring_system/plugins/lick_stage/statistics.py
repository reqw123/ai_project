"""Lick zone time/hit accumulation with bout tracking."""


_ZONES = ("BODY", "FL", "FR", "HL", "HR")

# Maps raw zone labels to statistics keys
_LABEL_TO_KEY = {
    "BODY_CENTER": "BODY",
    "FL":          "FL",
    "FR":          "FR",
    "HL":          "HL",
    "HR":          "HR",
}


def _stats_key(zone_label: str) -> str:
    return _LABEL_TO_KEY.get(zone_label, "")


class LickStatistics:
    """
    Accumulates per-zone lick time and hit counts with bout tracking.
    Not thread-safe; must be updated from a single thread.
    """

    def __init__(self):
        self._time:  dict = {z: 0.0 for z in _ZONES}
        self._hits:  dict = {z: 0   for z in _ZONES}
        # bout tracking
        self._bout_count: dict = {z: 0   for z in _ZONES}
        self._bout_sec:   dict = {z: 0.0 for z in _ZONES}
        self._active_bout_zone: str   = ""
        self._active_bout_sec:  float = 0.0
        self._prev_key: str = ""

    def update(self, zone_label: str, dt_sec: float) -> None:
        """
        Call once per frame.

        zone_label — raw label from find_nearest_zone (BODY_CENTER / FL / … / NO_TARGET)
        dt_sec     — elapsed seconds for this frame
        """
        key = _stats_key(zone_label)

        # Hit = transition *into* a lick zone
        if key and key != self._prev_key:
            self._hits[key] += 1

        if key and dt_sec > 0:
            self._time[key] += dt_sec
            # Bout accumulation
            if key == self._active_bout_zone:
                self._active_bout_sec += dt_sec
            else:
                # Close previous bout
                if self._active_bout_zone and self._active_bout_sec > 0:
                    self._bout_count[self._active_bout_zone] += 1
                    self._bout_sec[self._active_bout_zone]   += self._active_bout_sec
                self._active_bout_zone = key
                self._active_bout_sec  = dt_sec
        else:
            # No contact this frame — close any open bout
            if self._active_bout_zone and self._active_bout_sec > 0:
                self._bout_count[self._active_bout_zone] += 1
                self._bout_sec[self._active_bout_zone]   += self._active_bout_sec
            self._active_bout_zone = ""
            self._active_bout_sec  = 0.0

        self._prev_key = key

    def best_zone(self) -> str:
        """Return the zone with the highest accumulated lick time."""
        total = sum(self._time.values())
        if total <= 0:
            return "NO_TARGET"
        return max(self._time, key=self._time.get)

    def zone_stats(self, key: str):
        """Return (hits, time_sec) for the given statistics key."""
        return self._hits.get(key, 0), self._time.get(key, 0.0)

    def reset(self) -> None:
        for z in _ZONES:
            self._time[z]       = 0.0
            self._hits[z]       = 0
            self._bout_count[z] = 0
            self._bout_sec[z]   = 0.0
        self._active_bout_zone = ""
        self._active_bout_sec  = 0.0
        self._prev_key         = ""
