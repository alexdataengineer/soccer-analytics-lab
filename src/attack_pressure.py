"""
Attack and Pressure Detection Module
Detecção de ataques e pressão - momentos em que o time avança com múltiplos jogadores
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class AttackMoment:
    """Classe para representar um momento de ataque"""
    frame_start: int
    frame_end: int
    team: str
    player_count: int
    attack_intensity: float
    field_position: str  # 'defensive', 'midfield', 'attacking'
    duration: int
    timestamp: float


@dataclass
class PressureMoment:
    """Classe para representar um momento de pressão"""
    frame_start: int
    frame_end: int
    team: str
    pressure_intensity: float
    players_involved: List[str]
    target_area: str
    duration: int
    timestamp: float


class AttackPressureDetector:
    """Detector de ataques e pressão"""
    
    def __init__(self, 
                 attack_threshold: int = 5,
                 pressure_threshold: int = 5,
                 attack_duration_min: int = 15,
                 pressure_duration_min: int = 10):
        self.attack_threshold = attack_threshold
        self.pressure_threshold = pressure_threshold
        self.attack_duration_min = attack_duration_min
        self.pressure_duration_min = pressure_duration_min
        self.attacks = []
        self.pressure_moments = []
        
    def detect_attacks_and_pressure(self, tracking_data: List[Dict]) -> Dict:
        """
        Detecta ataques e momentos de pressão
        
        Args:
            tracking_data: Dados de tracking da partida
            
        Returns:
            Dict com ataques e pressão detectados
        """
        self.attacks = []
        self.pressure_moments = []
        
        # Analisar cada frame
        for frame_idx, frame in enumerate(tracking_data):
            if 'players' not in frame:
                continue
            
            # Detectar ataques
            self._detect_attack_in_frame(frame_idx, frame)
            
            # Detectar pressão
            self._detect_pressure_in_frame(frame_idx, frame)
        
        # Consolidar ataques e pressão consecutivos
        self._consolidate_attacks()
        self._consolidate_pressure()
        
        return {
            'attacks': self.attacks,
            'pressure_moments': self.pressure_moments,
            'statistics': self._calculate_statistics()
        }
    
    def _detect_attack_in_frame(self, frame_idx: int, frame: Dict):
        """Detecta ataques em um frame específico"""
        if 'players' not in frame:
            return
        
        # Agrupar jogadores por time
        team_positions = {}
        for player in frame['players']:
            team = player.get('team', 'unknown')
            if team not in team_positions:
                team_positions[team] = []
            team_positions[team].append({
                'id': player['id'],
                'position': player['position']
            })
        
        # Analisar cada time
        for team, players in team_positions.items():
            if len(players) < self.attack_threshold:
                continue
            
            # Calcular posição média do time
            avg_x = np.mean([p['position'][0] for p in players])
            avg_y = np.mean([p['position'][1] for p in players])
            
            # Determinar zona do campo
            field_zone = self._get_field_zone(avg_x, avg_y)
            
            # Calcular intensidade do ataque
            attack_intensity = self._calculate_attack_intensity(players, avg_x)
            
            # Verificar se é um ataque válido
            if attack_intensity > 0.6:  # Threshold de intensidade
                attack = AttackMoment(
                    frame_start=frame_idx,
                    frame_end=frame_idx,
                    team=team,
                    player_count=len(players),
                    attack_intensity=attack_intensity,
                    field_position=field_zone,
                    duration=1,
                    timestamp=frame_idx / 30.0
                )
                self.attacks.append(attack)
    
    def _detect_pressure_in_frame(self, frame_idx: int, frame: Dict):
        """Detecta pressão em um frame específico"""
        if 'players' not in frame or 'ball' not in frame or not frame['ball']:
            return
        
        ball_x, ball_y = frame['ball']
        
        # Agrupar jogadores por time
        team_positions = {}
        for player in frame['players']:
            team = player.get('team', 'unknown')
            if team not in team_positions:
                team_positions[team] = []
            team_positions[team].append({
                'id': player['id'],
                'position': player['position']
            })
        
        # Analisar pressão de cada time
        for team, players in team_positions.items():
            if len(players) < self.pressure_threshold:
                continue
            
            # Encontrar jogadores próximos à bola
            nearby_players = []
            for player in players:
                player_x, player_y = player['position']
                distance_to_ball = np.sqrt((ball_x - player_x)**2 + (ball_y - player_y)**2)
                
                if distance_to_ball <= 0.3:  # Raio de pressão
                    nearby_players.append({
                        'id': player['id'],
                        'distance': distance_to_ball,
                        'position': player['position']
                    })
            
            # Verificar se há pressão suficiente
            if len(nearby_players) >= 3:  # Mínimo 3 jogadores próximos
                pressure_intensity = self._calculate_pressure_intensity(nearby_players, ball_x, ball_y)
                
                if pressure_intensity > 0.5:  # Threshold de pressão
                    pressure = PressureMoment(
                        frame_start=frame_idx,
                        frame_end=frame_idx,
                        team=team,
                        pressure_intensity=pressure_intensity,
                        players_involved=[p['id'] for p in nearby_players],
                        target_area=self._get_field_zone(ball_x, ball_y),
                        duration=1,
                        timestamp=frame_idx / 30.0
                    )
                    self.pressure_moments.append(pressure)
    
    def _get_field_zone(self, x: float, y: float) -> str:
        """Determina a zona do campo baseada nas coordenadas"""
        if x < -0.2:
            return 'defensive'
        elif x > 0.2:
            return 'attacking'
        else:
            return 'midfield'
    
    def _calculate_attack_intensity(self, players: List[Dict], avg_x: float) -> float:
        """Calcula a intensidade do ataque"""
        if not players:
            return 0.0
        
        # Fatores para calcular intensidade:
        # 1. Posição média no campo (mais avançada = mais intenso)
        # 2. Concentração dos jogadores (mais próximos = mais intenso)
        # 3. Número de jogadores envolvidos
        
        # Fator 1: Posição no campo
        position_factor = (avg_x + 0.5) / 1.0  # Normalizar para 0-1
        
        # Fator 2: Concentração dos jogadores
        positions = np.array([p['position'] for p in players])
        distances = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        concentration_factor = 1.0 - np.mean(distances) if distances else 0.0
        
        # Fator 3: Número de jogadores
        player_factor = min(len(players) / 11.0, 1.0)  # Normalizar para 11 jogadores
        
        # Combinar fatores
        intensity = (position_factor * 0.4 + concentration_factor * 0.3 + player_factor * 0.3)
        
        return np.clip(intensity, 0, 1)
    
    def _calculate_pressure_intensity(self, nearby_players: List[Dict], ball_x: float, ball_y: float) -> float:
        """Calcula a intensidade da pressão"""
        if not nearby_players:
            return 0.0
        
        # Fatores para calcular pressão:
        # 1. Número de jogadores próximos
        # 2. Distância média à bola
        # 3. Posição da bola no campo
        
        # Fator 1: Número de jogadores
        player_factor = min(len(nearby_players) / 5.0, 1.0)
        
        # Fator 2: Distância média
        avg_distance = np.mean([p['distance'] for p in nearby_players])
        distance_factor = 1.0 - (avg_distance / 0.3)  # Normalizar para raio de 0.3
        
        # Fator 3: Posição da bola
        position_factor = abs(ball_x)  # Mais pressão nas extremidades
        
        # Combinar fatores
        intensity = (player_factor * 0.4 + distance_factor * 0.4 + position_factor * 0.2)
        
        return np.clip(intensity, 0, 1)
    
    def _consolidate_attacks(self):
        """Consolida ataques consecutivos"""
        if not self.attacks:
            return
        
        consolidated = []
        current_attack = None
        
        for attack in sorted(self.attacks, key=lambda x: x.frame_start):
            if current_attack is None:
                current_attack = attack
            elif (attack.team == current_attack.team and 
                  attack.frame_start == current_attack.frame_end + 1 and
                  attack.field_position == current_attack.field_position):
                # Continuar o ataque atual
                current_attack.frame_end = attack.frame_end
                current_attack.duration += 1
                current_attack.attack_intensity = max(current_attack.attack_intensity, 
                                                    attack.attack_intensity)
                current_attack.player_count = max(current_attack.player_count, 
                                                attack.player_count)
            else:
                # Finalizar ataque atual e iniciar novo
                if current_attack.duration >= self.attack_duration_min:
                    consolidated.append(current_attack)
                current_attack = attack
        
        # Adicionar último ataque
        if current_attack and current_attack.duration >= self.attack_duration_min:
            consolidated.append(current_attack)
        
        self.attacks = consolidated
    
    def _consolidate_pressure(self):
        """Consolida momentos de pressão consecutivos"""
        if not self.pressure_moments:
            return
        
        consolidated = []
        current_pressure = None
        
        for pressure in sorted(self.pressure_moments, key=lambda x: x.frame_start):
            if current_pressure is None:
                current_pressure = pressure
            elif (pressure.team == current_pressure.team and 
                  pressure.frame_start == current_pressure.frame_end + 1 and
                  pressure.target_area == current_pressure.target_area):
                # Continuar a pressão atual
                current_pressure.frame_end = pressure.frame_end
                current_pressure.duration += 1
                current_pressure.pressure_intensity = max(current_pressure.pressure_intensity, 
                                                        pressure.pressure_intensity)
                # Combinar jogadores envolvidos
                current_pressure.players_involved.extend(pressure.players_involved)
                current_pressure.players_involved = list(set(current_pressure.players_involved))
            else:
                # Finalizar pressão atual e iniciar nova
                if current_pressure.duration >= self.pressure_duration_min:
                    consolidated.append(current_pressure)
                current_pressure = pressure
        
        # Adicionar última pressão
        if current_pressure and current_pressure.duration >= self.pressure_duration_min:
            consolidated.append(current_pressure)
        
        self.pressure_moments = consolidated
    
    def _calculate_statistics(self) -> Dict:
        """Calcula estatísticas dos ataques e pressão"""
        stats = {
            'total_attacks': len(self.attacks),
            'total_pressure_moments': len(self.pressure_moments),
            'attack_statistics': {},
            'pressure_statistics': {}
        }
        
        # Estatísticas por time
        team_attacks = {}
        team_pressure = {}
        
        for attack in self.attacks:
            team = attack.team
            if team not in team_attacks:
                team_attacks[team] = {
                    'count': 0,
                    'total_duration': 0,
                    'avg_intensity': 0,
                    'field_zones': {'defensive': 0, 'midfield': 0, 'attacking': 0}
                }
            
            team_attacks[team]['count'] += 1
            team_attacks[team]['total_duration'] += attack.duration
            team_attacks[team]['field_zones'][attack.field_position] += 1
        
        for pressure in self.pressure_moments:
            team = pressure.team
            if team not in team_pressure:
                team_pressure[team] = {
                    'count': 0,
                    'total_duration': 0,
                    'avg_intensity': 0,
                    'target_areas': {'defensive': 0, 'midfield': 0, 'attacking': 0}
                }
            
            team_pressure[team]['count'] += 1
            team_pressure[team]['total_duration'] += pressure.duration
            team_pressure[team]['target_areas'][pressure.target_area] += 1
        
        # Calcular médias
        for team in team_attacks:
            if team_attacks[team]['count'] > 0:
                team_attacks[team]['avg_duration'] = team_attacks[team]['total_duration'] / team_attacks[team]['count']
                team_attacks[team]['avg_intensity'] = np.mean([a.attack_intensity for a in self.attacks if a.team == team])
        
        for team in team_pressure:
            if team_pressure[team]['count'] > 0:
                team_pressure[team]['avg_duration'] = team_pressure[team]['total_duration'] / team_pressure[team]['count']
                team_pressure[team]['avg_intensity'] = np.mean([p.pressure_intensity for p in self.pressure_moments if p.team == team])
        
        stats['attack_statistics'] = team_attacks
        stats['pressure_statistics'] = team_pressure
        
        return stats
    
    def create_momentum_chart(self, output_path: str = "outputs/match_momentum.png"):
        """Cria gráfico de momentum da partida"""
        if not self.attacks and not self.pressure_moments:
            return
        
        plt.figure(figsize=(15, 8))
        
        # Criar timeline
        max_frame = 0
        if self.attacks:
            max_frame = max(max_frame, max(a.frame_end for a in self.attacks))
        if self.pressure_moments:
            max_frame = max(max_frame, max(p.frame_end for p in self.pressure_moments))
        
        timeline = np.arange(0, max_frame + 1)
        momentum_team1 = np.zeros_like(timeline, dtype=float)
        momentum_team2 = np.zeros_like(timeline, dtype=float)
        
        # Adicionar ataques ao momentum
        for attack in self.attacks:
            team = attack.team
            for frame in range(attack.frame_start, attack.frame_end + 1):
                if frame < len(timeline):
                    if team == 'team1':
                        momentum_team1[frame] += attack.attack_intensity
                    else:
                        momentum_team2[frame] += attack.attack_intensity
        
        # Adicionar pressão ao momentum
        for pressure in self.pressure_moments:
            team = pressure.team
            for frame in range(pressure.frame_start, pressure.frame_end + 1):
                if frame < len(timeline):
                    if team == 'team1':
                        momentum_team1[frame] += pressure.pressure_intensity * 0.5
                    else:
                        momentum_team2[frame] += pressure.pressure_intensity * 0.5
        
        # Plotar momentum
        time_minutes = timeline / (30 * 60)  # Converter para minutos
        
        plt.plot(time_minutes, momentum_team1, label='Time 1', linewidth=2, color='red')
        plt.plot(time_minutes, momentum_team2, label='Time 2', linewidth=2, color='blue')
        
        # Adicionar áreas de ataques
        for attack in self.attacks:
            start_min = attack.frame_start / (30 * 60)
            end_min = attack.frame_end / (30 * 60)
            team_color = 'red' if attack.team == 'team1' else 'blue'
            plt.axvspan(start_min, end_min, alpha=0.2, color=team_color)
        
        plt.xlabel('Tempo (minutos)')
        plt.ylabel('Momentum')
        plt.title('Momentum da Partida - Ataques e Pressão')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_path: str = "outputs/attack_pressure_analysis.json"):
        """Salva os resultados da análise de ataques e pressão"""
        results = {
            'attacks': [
                {
                    'frame_start': a.frame_start,
                    'frame_end': a.frame_end,
                    'team': a.team,
                    'player_count': a.player_count,
                    'attack_intensity': a.attack_intensity,
                    'field_position': a.field_position,
                    'duration': a.duration,
                    'timestamp': a.timestamp
                }
                for a in self.attacks
            ],
            'pressure_moments': [
                {
                    'frame_start': p.frame_start,
                    'frame_end': p.frame_end,
                    'team': p.team,
                    'pressure_intensity': p.pressure_intensity,
                    'players_involved': p.players_involved,
                    'target_area': p.target_area,
                    'duration': p.duration,
                    'timestamp': p.timestamp
                }
                for p in self.pressure_moments
            ],
            'statistics': self._calculate_statistics()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2) 