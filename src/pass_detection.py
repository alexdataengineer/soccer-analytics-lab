"""
Pass Detection Module
Detecção automática de passes usando tracking da bola e posições dos jogadores
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Pass:
    """Classe para representar um passe"""
    frame_start: int
    frame_end: int
    player_from: str
    player_to: str
    start_position: Tuple[float, float]
    end_position: Tuple[float, float]
    distance: float
    duration: int
    success: bool
    team: str
    timestamp: float


class PassDetector:
    """Classe para detecção automática de passes"""
    
    def __init__(self, 
                 ball_proximity_threshold: float = 0.1,
                 pass_duration_threshold: int = 30,
                 min_pass_distance: float = 0.05):
        self.ball_proximity_threshold = ball_proximity_threshold
        self.pass_duration_threshold = pass_duration_threshold
        self.min_pass_distance = min_pass_distance
        self.passes = []
        self.player_positions = {}
        self.ball_positions = []
        
    def detect_passes(self, tracking_data: List[Dict]) -> List[Pass]:
        """
        Detecta passes nos dados de tracking
        
        Args:
            tracking_data: Lista de frames com posições dos jogadores e bola
            
        Returns:
            Lista de passes detectados
        """
        self.passes = []
        self.player_positions = {}
        self.ball_positions = []
        
        # Extrair posições dos jogadores e bola
        for frame_idx, frame in enumerate(tracking_data):
            # Posições dos jogadores
            if 'players' in frame:
                for player in frame['players']:
                    player_id = player['id']
                    if player_id not in self.player_positions:
                        self.player_positions[player_id] = []
                    self.player_positions[player_id].append({
                        'frame': frame_idx,
                        'position': player['position'],
                        'team': player.get('team', 'unknown')
                    })
            
            # Posição da bola
            if 'ball' in frame and frame['ball']:
                self.ball_positions.append({
                    'frame': frame_idx,
                    'position': frame['ball']
                })
        
        # Detectar passes
        self._find_pass_sequences()
        
        return self.passes
    
    def _find_pass_sequences(self):
        """Encontra sequências de passes"""
        if len(self.ball_positions) < 2:
            return
        
        # Para cada posição da bola, encontrar jogadores próximos
        ball_player_proximity = []
        
        for ball_pos in self.ball_positions:
            frame = ball_pos['frame']
            ball_x, ball_y = ball_pos['position']
            
            # Encontrar jogadores próximos à bola neste frame
            nearby_players = []
            for player_id, positions in self.player_positions.items():
                for pos_data in positions:
                    if pos_data['frame'] == frame:
                        player_x, player_y = pos_data['position']
                        distance = np.sqrt((ball_x - player_x)**2 + (ball_y - player_y)**2)
                        
                        if distance <= self.ball_proximity_threshold:
                            nearby_players.append({
                                'player_id': player_id,
                                'distance': distance,
                                'position': (player_x, player_y),
                                'team': pos_data['team']
                            })
            
            ball_player_proximity.append({
                'frame': frame,
                'ball_position': (ball_x, ball_y),
                'nearby_players': nearby_players
            })
        
        # Detectar sequências de passes
        self._identify_pass_sequences(ball_player_proximity)
    
    def _identify_pass_sequences(self, ball_player_proximity: List[Dict]):
        """Identifica sequências de passes"""
        if len(ball_player_proximity) < 2:
            return
        
        current_pass = None
        
        for i in range(len(ball_player_proximity)):
            current_frame = ball_player_proximity[i]
            
            # Se não há jogadores próximos à bola
            if not current_frame['nearby_players']:
                # Finalizar passe atual se existir
                if current_pass:
                    self._finalize_pass(current_pass, current_frame['frame'])
                    current_pass = None
                continue
            
            # Se há jogadores próximos à bola
            closest_player = min(current_frame['nearby_players'], 
                               key=lambda x: x['distance'])
            
            if current_pass is None:
                # Iniciar novo passe
                current_pass = {
                    'frame_start': current_frame['frame'],
                    'player_from': closest_player['player_id'],
                    'start_position': closest_player['position'],
                    'team': closest_player['team'],
                    'ball_positions': [current_frame['ball_position']]
                }
            else:
                # Continuar passe atual
                current_pass['ball_positions'].append(current_frame['ball_position'])
                
                # Verificar se o jogador mudou (passou para outro jogador)
                if closest_player['player_id'] != current_pass['player_from']:
                    # Verificar se é um passe válido
                    if self._is_valid_pass(current_pass, closest_player):
                        # Finalizar passe atual
                        self._finalize_pass(current_pass, current_frame['frame'], 
                                          closest_player)
                        
                        # Iniciar novo passe com o novo jogador
                        current_pass = {
                            'frame_start': current_frame['frame'],
                            'player_from': closest_player['player_id'],
                            'start_position': closest_player['position'],
                            'team': closest_player['team'],
                            'ball_positions': [current_frame['ball_position']]
                        }
    
    def _is_valid_pass(self, current_pass: Dict, receiving_player: Dict) -> bool:
        """Verifica se um passe é válido"""
        # Calcular duração do passe
        duration = len(current_pass['ball_positions'])
        
        # Calcular distância do passe
        start_pos = current_pass['start_position']
        end_pos = receiving_player['position']
        distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # Verificar critérios
        if duration > self.pass_duration_threshold:
            return False  # Passe muito longo (provavelmente não é um passe)
        
        if distance < self.min_pass_distance:
            return False  # Distância muito pequena
        
        # Verificar se a bola se moveu de forma contínua
        ball_positions = current_pass['ball_positions']
        if len(ball_positions) < 2:
            return False
        
        # Verificar se não há interrupções grandes no movimento da bola
        for i in range(1, len(ball_positions)):
            prev_pos = ball_positions[i-1]
            curr_pos = ball_positions[i]
            ball_distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            
            if ball_distance > 0.2:  # Movimento muito brusco da bola
                return False
        
        return True
    
    def _finalize_pass(self, current_pass: Dict, end_frame: int, 
                      receiving_player: Optional[Dict] = None):
        """Finaliza um passe detectado"""
        if not receiving_player:
            # Passe não completado
            return
        
        # Calcular métricas do passe
        duration = end_frame - current_pass['frame_start']
        start_pos = current_pass['start_position']
        end_pos = receiving_player['position']
        distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # Determinar sucesso do passe
        success = True  # Simplificado - assume que se chegou ao jogador, foi bem-sucedido
        
        # Criar objeto Pass
        pass_obj = Pass(
            frame_start=current_pass['frame_start'],
            frame_end=end_frame,
            player_from=current_pass['player_from'],
            player_to=receiving_player['player_id'],
            start_position=start_pos,
            end_position=end_pos,
            distance=distance,
            duration=duration,
            success=success,
            team=current_pass['team'],
            timestamp=end_frame / 30.0  # Assumindo 30 FPS
        )
        
        self.passes.append(pass_obj)
    
    def get_pass_statistics(self) -> Dict:
        """Calcula estatísticas dos passes detectados"""
        if not self.passes:
            return {}
        
        total_passes = len(self.passes)
        successful_passes = sum(1 for p in self.passes if p.success)
        success_rate = successful_passes / total_passes if total_passes > 0 else 0
        
        # Estatísticas por time
        team_stats = {}
        for pass_obj in self.passes:
            team = pass_obj.team
            if team not in team_stats:
                team_stats[team] = {
                    'total_passes': 0,
                    'successful_passes': 0,
                    'total_distance': 0,
                    'avg_distance': 0,
                    'passes_by_player': {}
                }
            
            team_stats[team]['total_passes'] += 1
            team_stats[team]['total_distance'] += pass_obj.distance
            
            if pass_obj.success:
                team_stats[team]['successful_passes'] += 1
            
            # Estatísticas por jogador
            for player_id in [pass_obj.player_from, pass_obj.player_to]:
                if player_id not in team_stats[team]['passes_by_player']:
                    team_stats[team]['passes_by_player'][player_id] = {
                        'passes_made': 0,
                        'passes_received': 0
                    }
            
            team_stats[team]['passes_by_player'][pass_obj.player_from]['passes_made'] += 1
            team_stats[team]['passes_by_player'][pass_obj.player_to]['passes_received'] += 1
        
        # Calcular médias
        for team in team_stats:
            if team_stats[team]['total_passes'] > 0:
                team_stats[team]['success_rate'] = team_stats[team]['successful_passes'] / team_stats[team]['total_passes']
                team_stats[team]['avg_distance'] = team_stats[team]['total_distance'] / team_stats[team]['total_passes']
        
        return {
            'total_passes': total_passes,
            'successful_passes': successful_passes,
            'success_rate': success_rate,
            'avg_pass_distance': np.mean([p.distance for p in self.passes]),
            'avg_pass_duration': np.mean([p.duration for p in self.passes]),
            'team_statistics': team_stats
        }
    
    def save_results(self, output_path: str = "outputs/pass_detection.json"):
        """Salva os resultados da detecção de passes"""
        results = {
            'passes': [
                {
                    'frame_start': p.frame_start,
                    'frame_end': p.frame_end,
                    'player_from': p.player_from,
                    'player_to': p.player_to,
                    'start_position': p.start_position,
                    'end_position': p.end_position,
                    'distance': p.distance,
                    'duration': p.duration,
                    'success': p.success,
                    'team': p.team,
                    'timestamp': p.timestamp
                }
                for p in self.passes
            ],
            'statistics': self.get_pass_statistics()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2) 