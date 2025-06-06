import time
from typing import Union
import numpy as np
from src.middlewares.slogger import SafeLogger
from src.funcs.base import emd_efecto, ABECEDARY
from src.middlewares.profile import profiler_manager, profile
from src.funcs.format import fmt_biparte_q
from src.controllers.manager import Manager
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import (
    QNODES_ANALYSIS_TAG,
    QNODES_LABEL,
    QNODES_STRAREGY_TAG,
)
from src.constants.base import (
    TYPE_TAG,
    NET_LABEL,
    INFTY_NEG,
    INFTY_POS,
    LAST_IDX,
    EFECTO,
    ACTUAL,
)
from concurrent.futures import ThreadPoolExecutor


class Geometric(SIA):
    """Clase Geometric para análisis mediante enfoque geométrico-topológico."""

    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.m: int
        self.n: int
        self.tiempos: tuple[np.ndarray, np.ndarray]
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.vertices: set[tuple]
        self.memoria_particiones = dict()
        
        self.indices_alcance: np.ndarray
        self.indices_mecanismo: np.ndarray
        
        self.logger = SafeLogger("geometric_strategy")

    # ===== MÉTODOS AUXILIARES BÁSICOS =====
    
    def construir_espacio_geometrico(self):
        """Construye la representación geométrica del subsistema como hipercubo."""
        total_variables = self.n + self.m
        num_estados = 2 ** total_variables
        vertices = []
        
        for i in range(num_estados):
            estado_binario = tuple(int(bit) for bit in format(i, f'0{total_variables}b'))
            vertices.append(estado_binario)
        
        coordenadas = {}
        for estado in vertices:
            coordenadas[estado] = np.array(estado, dtype=np.float64)
        
        adyacencias = {}
        for estado_i in vertices:
            adyacencias[estado_i] = []
            for estado_j in vertices:
                if estado_i != estado_j and self.calcular_distancia_hamming(estado_i, estado_j) == 1:
                    adyacencias[estado_i].append(estado_j)
        
        self.hipercubo = {
            'vertices': vertices,
            'adyacencias': adyacencias,
            'coordenadas': coordenadas,
            'dimensiones': total_variables,
            'num_estados': num_estados
        }
        
        self.vertices_geometricos = set()
        for idx in self.indices_mecanismo:
            self.vertices_geometricos.add((ACTUAL, idx))
        for idx in self.indices_alcance:
            self.vertices_geometricos.add((EFECTO, idx))
        
        self.logger.info(f"Espacio geométrico construido: {total_variables}D hipercubo con {num_estados} estados")
        return self.hipercubo

    def calcular_distancia_hamming(self, estado_i: tuple, estado_j: tuple) -> int:
        """Calcula la distancia de Hamming entre dos estados binarios."""
        if len(estado_i) != len(estado_j):
            raise ValueError(f"Los estados deben tener la misma longitud")
        return sum(bit_i != bit_j for bit_i, bit_j in zip(estado_i, estado_j))

    def calcular_factor_decrecimiento(self, estado_i: tuple, estado_j: tuple) -> float:
        """Calcula el factor de decrecimiento exponencial γ = 2^(-dH(i,j))."""
        distancia = self.calcular_distancia_hamming(estado_i, estado_j)
        return 2.0 ** (-distancia)

    def obtener_vecinos_adyacentes(self, estado: tuple) -> list:
        """Genera todos los vecinos adyacentes de un estado."""
        vecinos = []
        for i in range(len(estado)):
            nuevo_estado = list(estado)
            nuevo_estado[i] = 1 - nuevo_estado[i]
            vecinos.append(tuple(nuevo_estado))
        return vecinos

    def obtener_vecinos_camino_optimo(self, estado_i: tuple, estado_j: tuple) -> list:
        """Identifica vecinos de i que están en caminos óptimos hacia j."""
        if estado_i == estado_j:
            return []
        
        distancia_i_j = self.calcular_distancia_hamming(estado_i, estado_j)
        if distancia_i_j == 1:
            return []
        
        vecinos_adyacentes = self.obtener_vecinos_adyacentes(estado_i)
        vecinos_optimos = []
        distancia_optima = distancia_i_j - 1
        
        for vecino in vecinos_adyacentes:
            distancia_vecino_j = self.calcular_distancia_hamming(vecino, estado_j)
            if distancia_vecino_j == distancia_optima:
                vecinos_optimos.append(vecino)
        
        return vecinos_optimos

    # ===== MÉTODOS DE TENSORES =====
    
    def descomponer_en_tensores(self):
        """Descompone la dinámica del subsistema en tensores elementales."""
        if not hasattr(self, 'hipercubo'):
            self.construir_espacio_geometrico()
        
        self.tensores_elementales = {}
        tpm_estado_nodo = self._extraer_tpm_estado_nodo()
        
        for idx_variable in range(self.m):
            variable_idx = self.indices_alcance[idx_variable]
            tensor_variable = self._crear_tensor_variable(tpm_estado_nodo, idx_variable, variable_idx)
            
            nombre_variable = f"var_{variable_idx}"
            self.tensores_elementales[nombre_variable] = {
                'tensor': tensor_variable,
                'indice_original': variable_idx,
                'indice_en_alcance': idx_variable,
                'probabilidades_condicionales': self._extraer_probabilidades_condicionales(
                    tensor_variable, variable_idx
                )
            }
        
        self.logger.info(f"Descomposición tensorial completada: {len(self.tensores_elementales)} tensores")
        return self.tensores_elementales

    def _extraer_tpm_estado_nodo(self):
        """Extrae TPM en formato estado-nodo."""
        tpm_completa = self.sia_subsistema.tpm
        num_estados = 2 ** self.n
        tpm_estado_nodo = np.zeros((num_estados, self.m))
        
        for estado_t in range(num_estados):
            for idx_var in range(self.m):
                prob_var_0 = 0.0
                for estado_t_plus_1 in range(2 ** self.m):
                    estado_binario = self._indice_a_binario(estado_t_plus_1, self.m)
                    if estado_binario[idx_var] == 0:
                        prob_var_0 += tpm_completa[estado_t, estado_t_plus_1]
                tpm_estado_nodo[estado_t, idx_var] = prob_var_0
        
        return tpm_estado_nodo

    def _crear_tensor_variable(self, tpm_estado_nodo, idx_variable, variable_idx):
        """Crea tensor elemental para una variable específica."""
        num_estados = 2 ** self.n
        tensor = {}
        
        for estado_idx in range(num_estados):
            estado_binario = self._indice_a_binario(estado_idx, self.n)
            prob_0 = tpm_estado_nodo[estado_idx, idx_variable]
            prob_1 = 1.0 - prob_0
            
            tensor[estado_binario] = {
                'prob_0': prob_0,
                'prob_1': prob_1,
                'valor_principal': prob_0
            }
        
        return tensor

    def _extraer_probabilidades_condicionales(self, tensor_variable, variable_idx):
        """Extrae probabilidades condicionales del tensor."""
        probabilidades = {}
        for estado_binario, probs in tensor_variable.items():
            probabilidades[estado_binario] = probs['prob_0']
        return probabilidades

    def obtener_valor_tensor(self, estado: tuple, tensor: dict) -> float:
        """Extrae valor de probabilidad condicional del tensor."""
        if not tensor:
            raise ValueError("El tensor no puede ser None o vacío")
        
        if 'probabilidades_condicionales' in tensor:
            probabilidades = tensor['probabilidades_condicionales']
            if estado in probabilidades:
                return float(probabilidades[estado])
            else:
                raise KeyError(f"Estado {estado} no encontrado")
        
        elif 'tensor' in tensor:
            tensor_interno = tensor['tensor']
            if estado in tensor_interno:
                return float(tensor_interno[estado]['valor_principal'])
            else:
                raise KeyError(f"Estado {estado} no encontrado en tensor interno")
        
        else:
            if estado in tensor:
                return float(tensor[estado])
            else:
                raise KeyError(f"Estado {estado} no encontrado")

    def obtener_tensor_por_variable(self, variable_idx: int) -> dict:
        """Obtiene tensor elemental para una variable específica."""
        if not hasattr(self, 'tensores_elementales'):
            raise ValueError("Los tensores elementales no han sido inicializados")
        
        nombre_variable = f"var_{variable_idx}"
        if nombre_variable not in self.tensores_elementales:
            raise KeyError(f"No se encontró tensor para la variable {variable_idx}")
        
        return self.tensores_elementales[nombre_variable]

    # ===== MÉTODOS DE CONVERSIÓN =====
    
    def _indice_a_binario(self, indice, num_bits):
        """Convierte índice entero a representación binaria."""
        return tuple(int(bit) for bit in format(indice, f'0{num_bits}b'))

    def convertir_indice_a_estado_binario(self, indice: int, num_bits: int) -> tuple:
        """Convierte índice entero a estado binario."""
        if indice < 0:
            raise ValueError(f"El índice debe ser no negativo")
        if indice >= 2 ** num_bits:
            raise ValueError(f"El índice {indice} excede el rango para {num_bits} bits")
        return tuple(int(bit) for bit in format(indice, f'0{num_bits}b'))

    def convertir_estado_binario_a_indice(self, estado: tuple) -> int:
        """Convierte estado binario a índice entero."""
        if not estado:
            raise ValueError("El estado no puede estar vacío")
        
        for i, bit in enumerate(estado):
            if bit not in (0, 1):
                raise ValueError(f"Bit en posición {i} debe ser 0 o 1")
        
        indice = 0
        for bit in estado:
            indice = (indice << 1) | bit
        return indice

    # ===== MÉTODOS DE EVALUACIÓN =====
    
    def _evaluar_biparticion_individual(self, candidata):
        """Evalúa una bipartición individual."""
        particion_1 = candidata['particion_1']
        
        temporal = [[], []]
        for tiempo, indice in particion_1:
            temporal[tiempo].append(indice)
        
        dims_alcance = np.array(temporal[EFECTO], dtype=np.int8)
        dims_mecanismo = np.array(temporal[ACTUAL], dtype=np.int8)
        
        if len(dims_alcance) == 0 and len(dims_mecanismo) == 0:
            distribucion_vacia = np.zeros(len(self.sia_dists_marginales))
            if len(distribucion_vacia) > 0:
                distribucion_vacia[0] = 1.0
            return INFTY_POS, distribucion_vacia
        
        particion_resultado = self.sia_subsistema.bipartir(dims_alcance, dims_mecanismo)
        distribucion_marginal = particion_resultado.distribucion_marginal()
        perdida_emd = emd_efecto(distribucion_marginal, self.sia_dists_marginales)
        
        return perdida_emd, distribucion_marginal

    # ===== MÉTODO PRINCIPAL DE LA ESTRATEGIA (OPTIMIZADO) =====
    
    @profile(context={TYPE_TAG: QNODES_ANALYSIS_TAG})
    def aplicar_estrategia(self, condicion: str, alcance: str, mecanismo: str):
        """Método principal optimizado con logging detallado."""
        
        start_time = time.time()
        self.logger.info("=== INICIANDO ESTRATEGIA GEOMÉTRICA ===")
        
        # Preparación
        self.logger.info("Paso 1: Preparando subsistema...")
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
        
        futuro = tuple((EFECTO, idx_efecto) for idx_efecto in self.sia_subsistema.indices_ncubos)
        presente = tuple((ACTUAL, idx_actual) for idx_actual in self.sia_subsistema.dims_ncubos)
        
        self.m = self.sia_subsistema.indices_ncubos.size
        self.n = self.sia_subsistema.dims_ncubos.size
        self.indices_alcance = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos
        
        self.tiempos = (np.zeros(self.n, dtype=np.int8), np.zeros(self.m, dtype=np.int8))
        
        vertices = list(presente + futuro)
        self.vertices = set(presente + futuro)
        
        self.logger.info(f"Sistema: {self.n} presentes, {self.m} futuras, {len(vertices)} vértices total")
        
        try:
            # VERSIÓN SIMPLIFICADA - Solo usar heurísticas, no calcular todos los costos
            self.logger.info("Paso 2: Generando candidatas heurísticas...")
            
            candidatas = []
            
            # Estrategia 1: Marginalización simple (la más efectiva)
            for var_idx in self.indices_alcance:
                particion_1 = [(EFECTO, var_idx)]
                particion_2 = []
                
                for other_var in self.indices_alcance:
                    if other_var != var_idx:
                        particion_2.append((EFECTO, other_var))
                
                for var_presente in self.indices_mecanismo:
                    particion_2.append((ACTUAL, var_presente))
                
                candidatas.append({
                    'particion_1': particion_1,
                    'particion_2': particion_2,
                    'tipo': f'marginalizar_var_{var_idx}'
                })
            
            # Estrategia 2: División por mitades (si hay múltiples variables)
            if len(self.indices_alcance) > 1:
                mitad = len(self.indices_alcance) // 2
                particion_primera = [(EFECTO, idx) for idx in self.indices_alcance[:mitad]]
                particion_segunda = [(EFECTO, idx) for idx in self.indices_alcance[mitad:]]
                
                # Distribuir variables presentes
                mitad_presente = len(self.indices_mecanismo) // 2
                particion_primera.extend([(ACTUAL, idx) for idx in self.indices_mecanismo[:mitad_presente]])
                particion_segunda.extend([(ACTUAL, idx) for idx in self.indices_mecanismo[mitad_presente:]])
                
                candidatas.append({
                    'particion_1': particion_primera,
                    'particion_2': particion_segunda,
                    'tipo': 'division_mitades'
                })
            
            self.logger.info(f"Generadas {len(candidatas)} candidatas")
            
            # Paso 3: Evaluar candidatas directamente (sin tabla de costos)
            self.logger.info("Paso 3: Evaluando candidatas...")
            
            mejor_biparticion = None
            perdida_minima = INFTY_POS
            mejor_distribucion = None
            
            for i, candidata in enumerate(candidatas):
                self.logger.info(f"Evaluando candidata {i+1}/{len(candidatas)}: {candidata['tipo']}")
                
                try:
                    perdida, distribucion = self._evaluar_biparticion_individual(candidata)
                    self.logger.info(f"  -> Pérdida: {perdida:.6f}")
                    
                    if perdida < perdida_minima:
                        perdida_minima = perdida
                        mejor_biparticion = candidata
                        mejor_distribucion = distribucion
                        self.logger.info(f"  -> ¡Nueva mejor pérdida!")
                    
                    if abs(perdida) < 1e-10:
                        self.logger.info("¡Pérdida cero encontrada! Deteniendo.")
                        break
                        
                except Exception as e:
                    self.logger.warning(f"Error en candidata {i+1}: {str(e)}")
                    continue
            
            # Formatear resultado
            if mejor_biparticion:
                mip = tuple(mejor_biparticion['particion_1'])
                self.memoria_particiones[mip] = perdida_minima, mejor_distribucion
                fmt_mip = fmt_biparte_q(list(mip), self.nodes_complement(mip))
                
                tiempo_total = time.time() - start_time
                self.logger.info(f"=== COMPLETADO EN {tiempo_total:.2f}s ===")
                self.logger.info(f"Mejor: {mejor_biparticion['tipo']} con pérdida {perdida_minima:.6f}")
            else:
                # Fallback
                mip = tuple(vertices[:1]) if vertices else tuple()
                perdida_minima = INFTY_POS
                mejor_distribucion = np.zeros(len(self.sia_dists_marginales))
                if len(mejor_distribucion) > 0:
                    mejor_distribucion[0] = 1.0
                fmt_mip = fmt_biparte_q(list(mip), self.nodes_complement(mip))
                self.memoria_particiones[mip] = perdida_minima, mejor_distribucion
                
                tiempo_total = time.time() - start_time
                self.logger.warning(f"=== FALLBACK EN {tiempo_total:.2f}s ===")
            
            return Solution(
                estrategia="Geometric",
                perdida=perdida_minima,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=mejor_distribucion,
                tiempo_total=tiempo_total,
                particion=fmt_mip,
            )
            
        except Exception as e:
            tiempo_total = time.time() - start_time
            self.logger.error(f"ERROR GENERAL en {tiempo_total:.2f}s: {str(e)}")
            
            # Fallback de emergencia
            mip = tuple(vertices[:1]) if vertices else tuple()
            distribucion_fallback = np.zeros(len(self.sia_dists_marginales))
            if len(distribucion_fallback) > 0:
                distribucion_fallback[0] = 1.0
            fmt_mip = fmt_biparte_q(list(mip), self.nodes_complement(mip))
            self.memoria_particiones[mip] = INFTY_POS, distribucion_fallback
            
            return Solution(
                estrategia="Geometric",
                perdida=INFTY_POS,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=distribucion_fallback,
                tiempo_total=tiempo_total,
                particion=fmt_mip,
            )
        finally:
            if hasattr(self, 'hipercubo'):
                del self.hipercubo
                print("Espacio geométrico liberado")
            if hasattr(self, 'tensores_elementales'):
                del self.tensores_elementales
                print("Tensores elementales liberados")
            if hasattr(self, 'vertices_geometricos'):
                del self.vertices_geometricos
                print("Vértices geométricos liberados")
            if hasattr(self, 'memoria_particiones'):
                self.logger.info(f"Memoria de particiones: {len(self.memoria_particiones)} entradas")
                del self.memoria_particiones
                print("Memoria de particiones liberada")
            self.sia_limpiar_memoria()
            print("Memoria SIA limpiada")
            
    
    
    
    
    def nodes_complement(self, nodes: list[tuple[int, int]]):
        """Obtiene el complemento de nodos (compatibilidad con QNodes)."""
        return list(set(self.vertices) - set(nodes))
