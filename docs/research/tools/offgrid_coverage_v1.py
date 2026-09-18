"""Development-selected reserve margins for the dedicated-PV reference policy.

The immutable original coverage-v2 remains available with its original settings.
This version increases CO2/CH4 working reserves after two fresh development
cases exposed closing deficits; it never changes the mission's return targets.
"""

from docs.research.tools.feasible_mission_controller import (
    CoverageSettings,
    DailyCoverageController,
)


class OffgridCoverageController(DailyCoverageController):
    id = "daily-coverage-offgrid-v1"

    def __init__(self):
        super().__init__(CoverageSettings(co2_buffer_kg=100.0, ch4_buffer_kg=50.0))
