"""Lagerstyring for en lille webshop.

Modulet holder styr på varer, beregner rabatter og afgør, hvornår der skal
genbestilles hos leverandøren.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

MOMS = 0.25
STORKUNDE_GRAENSE = 100
SIKKERHEDSLAGER_DAGE = 7


@dataclass
class Vare:
    varenummer: str
    navn: str
    pris_ex_moms: float
    antal_paa_lager: int
    leveringstid_dage: int


def pris_med_moms(vare: Vare) -> float:
    """Returnér prisen inklusive 25 % moms, afrundet til hele øre."""
    return round(vare.pris_ex_moms * (1 + MOMS), 2)


def beregn_rabat(antal: int, er_medlem: bool) -> float:
    """Beregn rabatprocenten for en ordre.

    Storkunder, der køber mindst 100 stk., får 15 % rabat. Medlemmer af
    kundeklubben får 5 % på alle ordrer. Rabatterne lægges ikke sammen;
    kunden får altid den største.
    """
    rabatter = [0.0]
    if antal >= STORKUNDE_GRAENSE:
        rabatter.append(0.15)
    if er_medlem:
        rabatter.append(0.05)
    return max(rabatter)


def genbestillingspunkt(gennemsnitligt_salg_pr_dag: float, vare: Vare) -> int:
    """Hvor lavt lageret må komme, før der skal bestilles nye varer.

    Punktet dækker salget i hele leveringstiden plus et sikkerhedslager på
    syv dages salg, så der ikke opstår udsolgt, hvis leverandøren er forsinket.
    """
    dage = vare.leveringstid_dage + SIKKERHEDSLAGER_DAGE
    return int(gennemsnitligt_salg_pr_dag * dage + 0.5)


class Lager:
    """Samling af varer med metoder til salg og opfølgning."""

    def __init__(self) -> None:
        self.varer: dict[str, Vare] = {}
        self.salgslog: list[tuple[date, str, int]] = []

    def tilfoej(self, vare: Vare) -> None:
        self.varer[vare.varenummer] = vare

    def saelg(self, varenummer: str, antal: int) -> None:
        """Træk varer fra lageret. Fejler, hvis der ikke er nok på lager."""
        vare = self.varer[varenummer]
        if antal > vare.antal_paa_lager:
            raise ValueError(f"Kun {vare.antal_paa_lager} stk. af {vare.navn} på lager")
        vare.antal_paa_lager -= antal
        self.salgslog.append((date.today(), varenummer, antal))

    def varer_til_genbestilling(self, salg_pr_dag: dict[str, float]) -> list[Vare]:
        """Find de varer, hvor lageret er nået ned under genbestillingspunktet."""
        return [
            vare
            for vare in self.varer.values()
            if vare.antal_paa_lager <= genbestillingspunkt(salg_pr_dag.get(vare.varenummer, 0.0), vare)
        ]
