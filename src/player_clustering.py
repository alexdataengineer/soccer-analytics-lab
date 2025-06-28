"""
Player Clustering Module
Clusterização de jogadores por estilo de jogo usando KMeans
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json


class PlayerClustering:
    """Classe para clusterização de jogadores por estilo de jogo"""
    
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.player_features = {}
        self.cluster_labels = {}
        self.cluster_names = {
            0: "Box-to-Box",
            1: "Fixo/Defensivo", 
            2: "Explosivo/Atacante",
            3: "Criativo/Meio-campo"
        }
    
    def extract_player_features(self, tracking_data: List[Dict]) -> Dict:
        """
        Extrai features dos jogadores baseado nos dados de tracking
        
        Args:
            tracking_data: Lista de frames com posições dos jogadores
            
        Returns:
            Dict com features por jogador
        """
        player_stats = {}
        
        for frame in tracking_data:
            if 'players' not in frame:
                continue
                
            for player in frame['players']:
                player_id = player['id']
                x, y = player['position']
                
                if player_id not in player_stats:
                    player_stats[player_id] = {
                        'positions': [],
                        'velocities': [],
                        'distances': [],
                        'areas': [],
                        'team': player.get('team', 'unknown')
                    }
                
                player_stats[player_id]['positions'].append([x, y])
        
        # Calcular features para cada jogador
        for player_id, stats in player_stats.items():
            positions = np.array(stats['positions'])
            
            if len(positions) < 2:
                continue
                
            # Velocidade média
            velocities = []
            for i in range(1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[i-1])
                velocities.append(dist)
            
            avg_velocity = np.mean(velocities) if velocities else 0
            
            # Área coberta (convex hull)
            if len(positions) >= 3:
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(positions)
                    area_covered = hull.volume  # área em 2D
                except:
                    area_covered = 0
            else:
                area_covered = 0
            
            # Distância total percorrida
            total_distance = np.sum(velocities)
            
            # Posição média no campo
            avg_x = np.mean(positions[:, 0])
            avg_y = np.mean(positions[:, 1])
            
            # Variabilidade de movimento
            movement_variability = np.std(velocities) if velocities else 0
            
            # Tempo em diferentes zonas do campo
            time_attacking = np.sum(positions[:, 0] > 0.5) / len(positions)  # lado direito
            time_defending = np.sum(positions[:, 0] < -0.5) / len(positions)  # lado esquerdo
            
            self.player_features[player_id] = {
                'avg_velocity': avg_velocity,
                'area_covered': area_covered,
                'total_distance': total_distance,
                'avg_x': avg_x,
                'avg_y': avg_y,
                'movement_variability': movement_variability,
                'time_attacking': time_attacking,
                'time_defending': time_defending,
                'team': stats['team']
            }
    
    def cluster_players(self) -> Dict:
        """
        Executa a clusterização dos jogadores
        
        Returns:
            Dict com resultados da clusterização
        """
        if not self.player_features:
            return {}
        
        # Preparar dados para clustering
        feature_names = ['avg_velocity', 'area_covered', 'total_distance', 
                        'movement_variability', 'time_attacking', 'time_defending']
        
        X = []
        player_ids = []
        
        for player_id, features in self.player_features.items():
            player_ids.append(player_id)
            X.append([features[feature] for feature in feature_names])
        
        X = np.array(X)
        
        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Aplicar PCA para visualização
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Executar KMeans
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Mapear labels para jogadores
        for i, player_id in enumerate(player_ids):
            self.cluster_labels[player_id] = int(cluster_labels[i])
        
        # Calcular centroides dos clusters
        centroids = self.kmeans.cluster_centers_
        centroids_unscaled = self.scaler.inverse_transform(centroids)
        
        # Analisar características de cada cluster
        cluster_analysis = {}
        for cluster_id in range(self.n_clusters):
            cluster_players = [pid for pid, label in self.cluster_labels.items() 
                             if label == cluster_id]
            
            if cluster_players:
                cluster_features = [self.player_features[pid] for pid in cluster_players]
                
                cluster_analysis[cluster_id] = {
                    'player_count': len(cluster_players),
                    'players': cluster_players,
                    'avg_velocity': np.mean([f['avg_velocity'] for f in cluster_features]),
                    'avg_area': np.mean([f['area_covered'] for f in cluster_features]),
                    'avg_distance': np.mean([f['total_distance'] for f in cluster_features]),
                    'avg_attacking_time': np.mean([f['time_attacking'] for f in cluster_features]),
                    'avg_defending_time': np.mean([f['time_defending'] for f in cluster_features])
                }
        
        return {
            'cluster_labels': self.cluster_labels,
            'cluster_analysis': cluster_analysis,
            'feature_names': feature_names,
            'X_pca': X_pca.tolist(),
            'player_ids': player_ids,
            'cluster_names': self.cluster_names
        }
    
    def get_player_style(self, player_id: str) -> str:
        """Retorna o estilo de jogo de um jogador específico"""
        if player_id in self.cluster_labels:
            cluster_id = self.cluster_labels[player_id]
            return self.cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        return "Desconhecido"
    
    def visualize_clusters(self, output_path: str = "outputs/player_clusters.png"):
        """Visualiza os clusters de jogadores"""
        if not hasattr(self, 'X_pca') or not self.cluster_labels:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plotar clusters
        for cluster_id in range(self.n_clusters):
            cluster_points = []
            for i, player_id in enumerate(self.player_features.keys()):
                if self.cluster_labels.get(player_id) == cluster_id:
                    cluster_points.append(self.X_pca[i])
            
            if cluster_points:
                cluster_points = np.array(cluster_points)
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                           label=self.cluster_names[cluster_id], alpha=0.7, s=100)
        
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.title('Clusterização de Jogadores por Estilo de Jogo')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Adicionar anotações dos jogadores
        for i, player_id in enumerate(self.player_features.keys()):
            cluster_id = self.cluster_labels.get(player_id)
            if cluster_id is not None:
                plt.annotate(f"P{player_id}", 
                           (self.X_pca[i, 0], self.X_pca[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_path: str = "outputs/player_clustering.json"):
        """Salva os resultados da clusterização"""
        results = {
            'cluster_labels': self.cluster_labels,
            'player_features': self.player_features,
            'cluster_names': self.cluster_names
        }
        
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
        
        results = convert_numpy(results)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2) 