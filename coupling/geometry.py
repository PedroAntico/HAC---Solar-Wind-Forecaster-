"""
Módulo para cálculo de métricas geométricas de eventos solares.
"""
from dataclasses import dataclass
import numpy as np
from typing import Optional, Union
from dataclasses_json import dataclass_json
import warnings

@dataclass_json
@dataclass
class EventGeometry:
    """
    Geometria aproximada de um evento solar.

    Attributes
    ----------
    longitude_deg : float
        Longitude heliográfica (E/W), em graus, da região ativa / CME.
        Oeste = positivo (W), Leste = negativo (E), convenção SWPC.
    latitude_deg : float
        Latitude heliográfica em graus (-90 a 90).
    angular_width_deg : float
        Largura angular estimada da CME (0–360).
    halo_flag : bool
        True se evento é halo / quase halo.
    source_region : Optional[str]
        Identificação opcional da região ativa.
    """
    longitude_deg: float
    latitude_deg: float
    angular_width_deg: float
    halo_flag: bool = False
    source_region: Optional[str] = None
    
    def __post_init__(self):
        """Validação dos dados de entrada."""
        self._validate()
    
    def _validate(self):
        """Valida os parâmetros geométricos."""
        errors = []
        
        # Valida longitude
        if not (-180 <= self.longitude_deg <= 180):
            errors.append(f"Longitude {self.longitude_deg} fora do intervalo [-180, 180]")
        
        # Valida latitude
        if not (-90 <= self.latitude_deg <= 90):
            errors.append(f"Latitude {self.latitude_deg} fora do intervalo [-90, 90]")
        
        # Valida largura angular
        if not (0 <= self.angular_width_deg <= 360):
            errors.append(f"Largura angular {self.angular_width_deg} fora do intervalo [0, 360]")
        
        if errors:
            raise ValueError(f"Erros na geometria do evento: {', '.join(errors)}")


def central_meridian_distance(longitude_deg: float) -> float:
    """
    Distância ao meridiano central (CMD) em graus.
    
    Parameters
    ----------
    longitude_deg : float
        Longitude heliográfica em graus.
    
    Returns
    -------
    float
        Distância absoluta ao meridiano central (0-180 graus).
    
    Examples
    --------
    >>> central_meridian_distance(0)
    0.0
    >>> central_meridian_distance(90)
    90.0
    >>> central_meridian_distance(-45)
    45.0
    """
    if not isinstance(longitude_deg, (int, float)):
        raise TypeError(f"Longitude deve ser numérica, recebido {type(longitude_deg)}")
    
    return abs(float(longitude_deg))


def impact_angle_score(geom: EventGeometry, use_width_correction: bool = True) -> float:
    """
    Retorna um score geométrico entre 0 e 1 para o potencial
    de acoplamento, baseado apenas em geometria.
    
    Score = 1.0 → impacto central provável
    Score = 0.0 → impacto de flanco/extrema borda
    
    Parameters
    ----------
    geom : EventGeometry
        Geometria do evento solar.
    use_width_correction : bool
        Se True, aplica correção baseada na largura angular da CME.
    
    Returns
    -------
    float
        Score geométrico entre 0 e 1.
    """
    if not isinstance(geom, EventGeometry):
        raise TypeError(f"geom deve ser EventGeometry, recebido {type(geom)}")
    
    # Distância ao meridiano central
    cmd = central_meridian_distance(geom.longitude_deg)
    
    # Penalização por distância do meridiano central
    # Usando função cosseno suavizada para melhor resposta
    base_score = np.cos(np.radians(min(cmd, 90)))
    
    # Fator de correção para CMEs muito largas
    if use_width_correction:
        width_factor = min(1.0, geom.angular_width_deg / 180.0)
        # CMEs muito largas são menos sensíveis à posição
        width_correction = 0.8 + 0.2 * width_factor
    else:
        width_correction = 1.0
    
    # Bônus para eventos halo
    halo_bonus = 0.15 if geom.halo_flag else 0.0
    
    # Penalização por latitude muito alta (eventos polares)
    lat_penalty = min(1.0, abs(geom.latitude_deg) / 45.0)
    
    # Score final
    score = base_score * width_correction * (1.0 - 0.3 * lat_penalty) + halo_bonus
    
    # Garante limites [0, 1]
    return float(np.clip(score, 0.0, 1.0))


def calculate_flank_encounter_angle(geom: EventGeometry, 
                                   earth_position: float = 0.0) -> float:
    """
    Calcula o ângulo de encontro com o flanco da magnetosfera.
    
    Parameters
    ----------
    geom : EventGeometry
        Geometria do evento solar.
    earth_position : float
        Posição da Terra em longitude heliográfica (padrão 0.0).
    
    Returns
    -------
    float
        Ângulo de encontro em graus (0 = frontal, 90 = lateral).
    """
    # Diferença angular entre a direção do CME e a Terra
    angular_separation = abs(geom.longitude_deg - earth_position)
    
    # Considera a largura da CME
    effective_angle = max(0, angular_separation - geom.angular_width_deg / 2)
    
    return min(90.0, effective_angle)


if __name__ == "__main__":
    # Testes da função
    import doctest
    doctest.testmod(verbose=True)
    
    # Teste com exemplos
    geom1 = EventGeometry(longitude_deg=0, latitude_deg=0, 
                         angular_width_deg=60, halo_flag=False)
    print(f"Score geométrico (frontal): {impact_angle_score(geom1):.3f}")
    
    geom2 = EventGeometry(longitude_deg=80, latitude_deg=30, 
                         angular_width_deg=30, halo_flag=False)
    print(f"Score geométrico (flanco): {impact_angle_score(geom2):.3f}")
