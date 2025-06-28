"""
Expected Possession Value (EPV) Module
Modelo que estima o "valor" de cada posição no campo
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class EPVModel:
    """Modelo Expected Possession Value (EPV)"""
    
    def __init__(self, field_width: float = 1.0, field_height: float = 1.0):
        self.field_width = field_width
        self.field_height = field_height
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.field_value_grid = None
        
    def create_field_value_grid(self, resolution: int = 50):
        """
        Cria uma grade de valores para o campo baseada em heurísticas
        
        Args:
            resolution: Resolução da grade (50x50 por padrão)
        """
        x = np.linspace(-self.field_width/2, self.field_width/2, resolution)
        y = np.linspace(-self.field_height/2, self.field_height/2, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Criar valores baseados em heurísticas de futebol
        values = np.zeros_like(X)
        
        for i in range(resolution):
            for j in range(resolution):
                x_pos = X[i, j]
                y_pos = Y[i, j]
                
                # Valor baseado na distância do gol
                # Gol direito (x = 0.5, y = 0)
                distance_to_right_goal = np.sqrt((x_pos - 0.5)**2 + y_pos**2)
                # Gol esquerdo (x = -0.5, y = 0)
                distance_to_left_goal = np.sqrt((x_pos + 0.5)**2 + y_pos**2)
                
                # Valor mais alto quanto mais próximo do gol
                right_goal_value = 1.0 / (1.0 + distance_to_right_goal)
                left_goal_value = 1.0 / (1.0 + distance_to_left_goal)
                
                # Valor baseado na posição no campo
                # Centro do campo tem valor médio
                center_distance = np.sqrt(x_pos**2 + y_pos**2)
                center_value = 0.5 * np.exp(-center_distance)
                
                # Valor baseado na proximidade das laterais
                side_distance = min(abs(y_pos - 0.5), abs(y_pos + 0.5))
                side_value = 0.3 * np.exp(-side_distance)
                
                # Combinar valores
                total_value = (right_goal_value + left_goal_value + center_value + side_value) / 4
                
                # Normalizar para 0-1
                values[i, j] = np.clip(total_value, 0, 1)
        
        self.field_value_grid = {
            'X': X.tolist(),
            'Y': Y.tolist(),
            'values': values.tolist(),
            'resolution': resolution
        }
    
    def get_position_value(self, x: float, y: float) -> float:
        """
        Obtém o valor de uma posição específica no campo
        
        Args:
            x, y: Coordenadas da posição
            
        Returns:
            Valor da posição (0-1)
        """
        if self.field_value_grid is None:
            self.create_field_value_grid()
        
        # Encontrar a célula mais próxima na grade
        x_grid = np.array(self.field_value_grid['X'])
        y_grid = np.array(self.field_value_grid['Y'])
        values = np.array(self.field_value_grid['values'])
        
        # Encontrar índices mais próximos
        x_idx = np.argmin(np.abs(x_grid[0, :] - x))
        y_idx = np.argmin(np.abs(y_grid[:, 0] - y))
        
        return values[y_idx, x_idx]
    
    def calculate_possession_value(self, tracking_data: List[Dict]) -> Dict:
        """
        Calcula o valor de posse para cada frame
        
        Args:
            tracking_data: Dados de tracking da partida
            
        Returns:
            Dict com valores de posse por frame
        """
        if self.field_value_grid is None:
            self.create_field_value_grid()
        
        possession_values = []
        
        for frame_idx, frame in enumerate(tracking_data):
            frame_value = {
                'frame': frame_idx,
                'timestamp': frame_idx / 30.0,  # Assumindo 30 FPS
                'ball_value': 0.0,
                'team_possession_values': {},
                'total_value': 0.0
            }
            
            # Valor da posição da bola
            if 'ball' in frame and frame['ball']:
                ball_x, ball_y = frame['ball']
                ball_value = self.get_position_value(ball_x, ball_y)
                frame_value['ball_value'] = ball_value
                frame_value['total_value'] += ball_value
            
            # Valores por time baseado na posição dos jogadores
            if 'players' in frame:
                team_positions = {}
                
                for player in frame['players']:
                    team = player.get('team', 'unknown')
                    if team not in team_positions:
                        team_positions[team] = []
                    team_positions[team].append(player['position'])
                
                # Calcular valor médio por time
                for team, positions in team_positions.items():
                    team_values = []
                    for pos in positions:
                        x, y = pos
                        pos_value = self.get_position_value(x, y)
                        team_values.append(pos_value)
                    
                    avg_team_value = np.mean(team_values) if team_values else 0.0
                    frame_value['team_possession_values'][team] = avg_team_value
                    frame_value['total_value'] += avg_team_value
        
        return possession_values
    
    def train_model_from_data(self, training_data: List[Dict]):
        """
        Treina o modelo EPV com dados de treinamento
        (Para versão futura com dados reais de gols)
        
        Args:
            training_data: Dados de treinamento com posições e resultados
        """
        # Esta é uma implementação simplificada
        # Em uma versão real, você teria dados de gols marcados
        # e treinaria o modelo para prever a probabilidade de gol
        
        features = []
        targets = []
        
        for match_data in training_data:
            # Extrair features (posição da bola, posições dos jogadores, etc.)
            # e targets (se houve gol ou não)
            pass
        
        if features and targets:
            X = np.array(features)
            y = np.array(targets)
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
    
    def predict_goal_probability(self, frame_data: Dict) -> float:
        """
        Prediz a probabilidade de gol para um frame específico
        
        Args:
            frame_data: Dados do frame atual
            
        Returns:
            Probabilidade de gol (0-1)
        """
        if not self.is_trained:
            # Retornar valor baseado em heurísticas
            if 'ball' in frame_data and frame_data['ball']:
                ball_x, ball_y = frame_data['ball']
                return self.get_position_value(ball_x, ball_y)
            return 0.0
        
        # Extrair features do frame
        features = self._extract_frame_features(frame_data)
        
        if features:
            features_scaled = self.scaler.transform([features])
            return self.model.predict(features_scaled)[0]
        
        return 0.0
    
    def _extract_frame_features(self, frame_data: Dict) -> List[float]:
        """Extrai features de um frame para predição"""
        features = []
        
        # Posição da bola
        if 'ball' in frame_data and frame_data['ball']:
            ball_x, ball_y = frame_data['ball']
            features.extend([ball_x, ball_y])
        else:
            features.extend([0.0, 0.0])
        
        # Posições dos jogadores por time
        if 'players' in frame_data:
            team_positions = {}
            for player in frame_data['players']:
                team = player.get('team', 'unknown')
                if team not in team_positions:
                    team_positions[team] = []
                team_positions[team].append(player['position'])
            
            # Features para cada time (máximo 2 times)
            for team in list(team_positions.keys())[:2]:
                positions = team_positions[team]
                if positions:
                    avg_x = np.mean([pos[0] for pos in positions])
                    avg_y = np.mean([pos[1] for pos in positions])
                    features.extend([avg_x, avg_y])
                else:
                    features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return features
    
    def calculate_match_epv(self, tracking_data: List[Dict]) -> Dict:
        """
        Calcula EPV para toda a partida
        
        Args:
            tracking_data: Dados de tracking da partida
            
        Returns:
            Dict com análise EPV da partida
        """
        frame_values = []
        cumulative_values = {'team1': 0.0, 'team2': 0.0}
        
        for frame_idx, frame in enumerate(tracking_data):
            # Calcular valor do frame
            ball_value = 0.0
            if 'ball' in frame and frame['ball']:
                ball_x, ball_y = frame['ball']
                ball_value = self.get_position_value(ball_x, ball_y)
            
            # Determinar qual time tem a posse (baseado na proximidade da bola)
            possession_team = None
            if 'players' in frame and 'ball' in frame and frame['ball']:
                ball_x, ball_y = frame['ball']
                min_distance = float('inf')
                
                for player in frame['players']:
                    player_x, player_y = player['position']
                    distance = np.sqrt((ball_x - player_x)**2 + (ball_y - player_y)**2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        possession_team = player.get('team', 'unknown')
            
            # Acumular valores
            if possession_team:
                if possession_team not in cumulative_values:
                    cumulative_values[possession_team] = 0.0
                cumulative_values[possession_team] += ball_value
            
            frame_values.append({
                'frame': frame_idx,
                'timestamp': frame_idx / 30.0,
                'ball_value': ball_value,
                'possession_team': possession_team,
                'cumulative_team1': cumulative_values.get('team1', 0.0),
                'cumulative_team2': cumulative_values.get('team2', 0.0)
            })
        
        # Calcular estatísticas
        total_frames = len(frame_values)
        team1_frames = sum(1 for f in frame_values if f['possession_team'] == 'team1')
        team2_frames = sum(1 for f in frame_values if f['possession_team'] == 'team2')
        
        return {
            'frame_values': frame_values,
            'total_frames': total_frames,
            'team1_possession_frames': team1_frames,
            'team2_possession_frames': team2_frames,
            'team1_possession_percentage': team1_frames / total_frames if total_frames > 0 else 0,
            'team2_possession_percentage': team2_frames / total_frames if total_frames > 0 else 0,
            'team1_epv': cumulative_values.get('team1', 0.0),
            'team2_epv': cumulative_values.get('team2', 0.0),
            'total_epv': sum(cumulative_values.values())
        }
    
    def visualize_field_values(self, output_path: str = "outputs/epv_field_heatmap.png"):
        """Visualiza o mapa de calor dos valores do campo"""
        if self.field_value_grid is None:
            self.create_field_value_grid()
        
        plt.figure(figsize=(12, 8))
        
        X = np.array(self.field_value_grid['X'])
        Y = np.array(self.field_value_grid['Y'])
        values = np.array(self.field_value_grid['values'])
        
        # Criar mapa de calor
        heatmap = plt.imshow(values, cmap='RdYlGn', extent=[-0.5, 0.5, -0.5, 0.5], 
                           aspect='equal', origin='lower')
        
        # Adicionar contornos do campo
        plt.plot([-0.5, 0.5], [0, 0], 'k-', linewidth=2)  # Linha central
        plt.plot([0, 0], [-0.5, 0.5], 'k-', linewidth=2)  # Linha central vertical
        
        # Gols
        plt.plot([0.5, 0.5], [-0.1, 0.1], 'k-', linewidth=3)  # Gol direito
        plt.plot([-0.5, -0.5], [-0.1, 0.1], 'k-', linewidth=3)  # Gol esquerdo
        
        plt.colorbar(heatmap, label='Expected Possession Value')
        plt.title('Mapa de Calor - Expected Possession Value (EPV)')
        plt.xlabel('Posição X')
        plt.ylabel('Posição Y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, epv_data: Dict, output_path: str = "outputs/epv_analysis.json"):
        """Salva os resultados da análise EPV"""
        # Converter numpy types para Python nativos
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        epv_data = convert_numpy(epv_data)
        
        with open(output_path, 'w') as f:
            json.dump(epv_data, f, indent=2) 