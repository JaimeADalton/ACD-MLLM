#!/usr/bin/env python3
"""
ACD-MLLM CLI: Algoritmo de Convergencia Deliberativa Multi-LLM
Con persistencia JSONL completa para auditoría y reproducibilidad.

Uso:
    acd-mllm run "¿Pregunta?" --models gpt-4 claude gemini --output results/
    acd-mllm replay results/run_20240101_123456.jsonl --analyze
    acd-mllm benchmark questions.txt --models gpt-4 claude --iterations 5
"""

import asyncio
import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import asdict
import logging

# Importar clases del módulo principal
from acd_mllm_consensus import (
    run_acd_mllm_consensus, 
    create_model_ensemble,
    ConsensusResult,
    ConsensusType,
    LLMInterface,
    OpenAIInterface,
    AnthropicInterface, 
    GoogleInterface
)

logger = logging.getLogger(__name__)

class PersistenceManager:
    """Gestor de persistencia con formato JSONL para máxima compatibilidad"""
    
    def __init__(self, output_dir: str = "runs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.current_run_id = None
        self.current_file = None
        
    def start_run(self, question: str, models: List[str], config: Dict[str, Any] = None) -> str:
        """Inicia una nueva ejecución y crea archivo JSONL"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_id = f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        self.current_file = self.output_dir / f"{self.current_run_id}.jsonl"
        
        # Registro inicial con metadatos
        initial_record = {
            "type": "run_start",
            "run_id": self.current_run_id,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "models": models,
            "config": config or {},
            "version": "1.0"
        }
        
        self._append_record(initial_record)
        logger.info(f"📁 Nueva ejecución iniciada: {self.current_run_id}")
        
        return self.current_run_id
    
    def log_iteration(self, iteration_data: Dict[str, Any]):
        """Registra datos de una iteración específica"""
        
        record = {
            "type": "iteration",
            "run_id": self.current_run_id,
            "timestamp": datetime.now().isoformat(),
            **iteration_data
        }
        
        self._append_record(record)
    
    def log_model_interaction(self, model_name: str, interaction_type: str, 
                            prompt: str = None, response: str = None, 
                            metadata: Dict[str, Any] = None):
        """Registra interacciones individuales con modelos"""
        
        record = {
            "type": "model_interaction",
            "run_id": self.current_run_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "interaction_type": interaction_type,  # "generation", "evaluation", "synthesis"
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {}
        }
        
        self._append_record(record)
    
    def log_mechanism(self, mechanism_type: str, details: Dict[str, Any]):
        """Registra uso de mecanismos de desbloqueo"""
        
        record = {
            "type": "unlock_mechanism",
            "run_id": self.current_run_id,
            "timestamp": datetime.now().isoformat(),
            "mechanism_type": mechanism_type,
            "details": details
        }
        
        self._append_record(record)
    
    def finish_run(self, result: ConsensusResult, duration_seconds: float):
        """Finaliza la ejecución y registra resultado final"""
        
        # Convertir result a diccionario serializable
        result_dict = asdict(result)
        
        final_record = {
            "type": "run_complete",
            "run_id": self.current_run_id,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration_seconds,
            "result": result_dict
        }
        
        self._append_record(final_record)
        logger.info(f"✅ Ejecución completada: {self.current_file}")
        
        return self.current_file
    
    def _append_record(self, record: Dict[str, Any]):
        """Añade un registro al archivo JSONL actual"""
        
        if self.current_file:
            with open(self.current_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

class RunAnalyzer:
    """Analizador de ejecuciones guardadas en JSONL"""
    
    def __init__(self, jsonl_file: Path):
        self.jsonl_file = jsonl_file
        self.records = self._load_records()
        
    def _load_records(self) -> List[Dict[str, Any]]:
        """Carga todos los registros del archivo JSONL"""
        
        records = []
        try:
            with open(self.jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            return records
        except Exception as e:
            logger.error(f"Error cargando {self.jsonl_file}: {e}")
            return []
    
    def analyze(self) -> Dict[str, Any]:
        """Análisis completo de la ejecución"""
        
        analysis = {
            "run_metadata": self._get_run_metadata(),
            "consensus_analysis": self._analyze_consensus(),
            "model_performance": self._analyze_model_performance(),
            "mechanism_usage": self._analyze_mechanisms(),
            "timeline": self._build_timeline(),
            "quality_progression": self._analyze_quality_progression()
        }
        
        return analysis
    
    def _get_run_metadata(self) -> Dict[str, Any]:
        """Extrae metadatos básicos de la ejecución"""
        
        start_record = next((r for r in self.records if r["type"] == "run_start"), {})
        end_record = next((r for r in self.records if r["type"] == "run_complete"), {})
        
        return {
            "run_id": start_record.get("run_id", "unknown"),
            "question": start_record.get("question", ""),
            "models": start_record.get("models", []),
            "duration": end_record.get("duration_seconds", 0),
            "consensus_type": end_record.get("result", {}).get("consensus_type", "unknown"),
            "final_quality": end_record.get("result", {}).get("final_quality_score", 0),
            "iterations": end_record.get("result", {}).get("iterations_taken", 0)
        }
    
    def _analyze_consensus(self) -> Dict[str, Any]:
        """Analiza el proceso de consenso"""
        
        iterations = [r for r in self.records if r["type"] == "iteration"]
        
        consensus_evolution = []
        for iteration in iterations:
            consensus_evolution.append({
                "iteration": iteration.get("iteration", 0),
                "consensus_votes": iteration.get("consensus_votes", 0),
                "average_quality": iteration.get("average_quality", 0),
                "convergence_score": iteration.get("convergence_metrics", {}).get("similarity", 0)
            })
        
        return {
            "total_iterations": len(iterations),
            "consensus_evolution": consensus_evolution,
            "convergence_trend": self._calculate_convergence_trend(consensus_evolution)
        }
    
    def _analyze_model_performance(self) -> Dict[str, Any]:
        """Analiza rendimiento individual de modelos"""
        
        interactions = [r for r in self.records if r["type"] == "model_interaction"]
        
        model_stats = {}
        for interaction in interactions:
            model = interaction.get("model_name", "unknown")
            if model not in model_stats:
                model_stats[model] = {
                    "total_interactions": 0,
                    "generation_count": 0,
                    "evaluation_count": 0,
                    "synthesis_count": 0,
                    "avg_response_length": 0,
                    "response_lengths": []
                }
            
            model_stats[model]["total_interactions"] += 1
            interaction_type = interaction.get("interaction_type", "")
            
            if interaction_type == "generation":
                model_stats[model]["generation_count"] += 1
            elif interaction_type == "evaluation":
                model_stats[model]["evaluation_count"] += 1
            elif interaction_type == "synthesis":
                model_stats[model]["synthesis_count"] += 1
            
            response = interaction.get("response", "")
            if response:
                length = len(response)
                model_stats[model]["response_lengths"].append(length)
        
        # Calcular promedios
        for model, stats in model_stats.items():
            if stats["response_lengths"]:
                stats["avg_response_length"] = sum(stats["response_lengths"]) / len(stats["response_lengths"])
        
        return model_stats
    
    def _analyze_mechanisms(self) -> Dict[str, Any]:
        """Analiza uso de mecanismos de desbloqueo"""
        
        mechanisms = [r for r in self.records if r["type"] == "unlock_mechanism"]
        
        mechanism_usage = {}
        for mech in mechanisms:
            mech_type = mech.get("mechanism_type", "unknown")
            if mech_type not in mechanism_usage:
                mechanism_usage[mech_type] = 0
            mechanism_usage[mech_type] += 1
        
        return {
            "total_mechanisms_used": len(mechanisms),
            "mechanism_breakdown": mechanism_usage,
            "mechanisms_timeline": [
                {
                    "timestamp": m.get("timestamp"),
                    "mechanism": m.get("mechanism_type"),
                    "details": m.get("details", {})
                }
                for m in mechanisms
            ]
        }
    
    def _build_timeline(self) -> List[Dict[str, Any]]:
        """Construye timeline cronológico de eventos"""
        
        timeline_events = []
        
        for record in self.records:
            event = {
                "timestamp": record.get("timestamp"),
                "type": record["type"],
                "summary": self._summarize_record(record)
            }
            timeline_events.append(event)
        
        return sorted(timeline_events, key=lambda x: x["timestamp"])
    
    def _analyze_quality_progression(self) -> Dict[str, Any]:
        """Analiza progresión de calidad a lo largo del tiempo"""
        
        iterations = [r for r in self.records if r["type"] == "iteration"]
        
        quality_progression = []
        for iteration in iterations:
            quality_progression.append({
                "iteration": iteration.get("iteration", 0),
                "quality": iteration.get("average_quality", 0),
                "convergence": iteration.get("convergence_metrics", {}).get("similarity", 0)
            })
        
        # Calcular tendencias
        qualities = [q["quality"] for q in quality_progression]
        convergences = [c["convergence"] for c in quality_progression]
        
        return {
            "quality_progression": quality_progression,
            "quality_trend": self._calculate_trend(qualities),
            "convergence_trend": self._calculate_trend(convergences),
            "final_improvement": qualities[-1] - qualities[0] if len(qualities) > 1 else 0
        }
    
    def _calculate_convergence_trend(self, evolution: List[Dict]) -> str:
        """Calcula tendencia de convergencia"""
        
        if len(evolution) < 2:
            return "insufficient_data"
        
        scores = [e["convergence_score"] for e in evolution]
        trend = self._calculate_trend(scores)
        
        return trend
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calcula tendencia general de una serie de valores"""
        
        if len(values) < 2:
            return "insufficient_data"
        
        improvements = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
        degradations = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
        
        if improvements > degradations * 1.5:
            return "improving"
        elif degradations > improvements * 1.5:
            return "degrading"
        else:
            return "stable"
    
    def _summarize_record(self, record: Dict[str, Any]) -> str:
        """Crea resumen legible de un registro"""
        
        record_type = record["type"]
        
        if record_type == "run_start":
            return f"Iniciada ejecución con modelos: {', '.join(record.get('models', []))}"
        elif record_type == "iteration":
            return f"Iteración {record.get('iteration', 0)}: {record.get('consensus_votes', 0)} votos, calidad {record.get('average_quality', 0):.3f}"
        elif record_type == "model_interaction":
            return f"{record.get('model_name', 'Unknown')} - {record.get('interaction_type', 'interaction')}"
        elif record_type == "unlock_mechanism":
            return f"Mecanismo aplicado: {record.get('mechanism_type', 'unknown')}"
        elif record_type == "run_complete":
            result = record.get('result', {})
            return f"Completada: {result.get('consensus_type', 'unknown')} en {record.get('duration_seconds', 0):.1f}s"
        else:
            return f"Evento: {record_type}"

class ACDMLLMWithPersistence:
    """Wrapper del algoritmo principal con capacidades de persistencia"""
    
    def __init__(self, persistence_manager: PersistenceManager):
        self.persistence = persistence_manager
    
    async def run_with_logging(self, question: str, models: List[LLMInterface], 
                              config: Dict[str, Any] = None) -> ConsensusResult:
        """Ejecuta el algoritmo con logging completo"""
        
        # Iniciar run
        model_names = [m.model_name for m in models]
        run_id = self.persistence.start_run(question, model_names, config)
        
        # Ejecutar algoritmo con hooks de logging
        start_time = time.time()
        
        try:
            result = await run_acd_mllm_consensus(question, models)
            
            # Log resultado
            duration = time.time() - start_time
            self.persistence.finish_run(result, duration)
            
            return result
            
        except Exception as e:
            # Log error
            error_record = {
                "type": "error",
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "duration": time.time() - start_time
            }
            
            self.persistence._append_record(error_record)
            raise

def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Configura logging para CLI"""
    
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def create_models_from_args(model_names: List[str], config_file: Optional[str] = None) -> List[LLMInterface]:
    """Crea modelos basado en argumentos CLI"""
    
    # Cargar configuración de archivo si se proporciona
    config = {}
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    
    # Usar variables de entorno como fallback
    if not config:
        config = {
            'openai_key': os.getenv('OPENAI_API_KEY'),
            'anthropic_key': os.getenv('ANTHROPIC_API_KEY'),
            'google_key': os.getenv('GOOGLE_API_KEY'),
        }
    
    models = []
    model_mapping = {
        'gpt-4': lambda: OpenAIInterface("gpt-4", config.get('openai_key')),
        'gpt-3.5': lambda: OpenAIInterface("gpt-3.5-turbo", config.get('openai_key')),
        'claude': lambda: AnthropicInterface("claude-3-sonnet-20240229", config.get('anthropic_key')),
        'gemini': lambda: GoogleInterface("gemini-pro", config.get('google_key')),
    }
    
    for model_name in model_names:
        if model_name in model_mapping:
            try:
                model = model_mapping[model_name]()
                models.append(model)
                logger.info(f"✅ Configurado modelo: {model_name}")
            except Exception as e:
                logger.error(f"❌ Error configurando {model_name}: {e}")
        else:
            logger.warning(f"⚠️ Modelo desconocido: {model_name}")
    
    return models

async def run_consensus_command(args):
    """Comando principal: ejecutar consenso"""
    
    print("🚀 ACD-MLLM: Ejecutando algoritmo de consenso")
    print("=" * 60)
    
    # Crear modelos
    models = create_models_from_args(args.models, args.config)
    
    if not models:
        print("❌ No se pudieron configurar modelos. Verifica las claves API.")
        return 1
    
    # Configurar persistencia
    persistence = PersistenceManager(args.output)
    runner = ACDMLLMWithPersistence(persistence)
    
    # Configuración adicional
    config = {
        "max_iterations": args.max_iterations,
        "quality_threshold": args.quality_threshold,
        "similarity_threshold": args.similarity_threshold
    }
    
    print(f"❓ Pregunta: {args.question}")
    print(f"🤖 Modelos: {[m.model_name for m in models]}")
    print(f"📁 Output: {args.output}/")
    print("=" * 60)
    
    # Ejecutar
    start_time = time.time()
    try:
        result = await runner.run_with_logging(args.question, models, config)
        end_time = time.time()
        
        # Mostrar resultados
        print(f"\n🏆 RESULTADO DEL CONSENSO")
        print("=" * 40)
        print(f"✅ Tipo: {result.consensus_type.value.upper()}")
        print(f"📈 Iteraciones: {result.iterations_taken}")
        print(f"🎯 Convergencia: {result.convergence_score:.3f}")
        print(f"⭐ Calidad: {result.final_quality_score:.3f}")
        print(f"🔒 Confianza: {result.confidence_level:.3f}")
        print(f"⏱️ Tiempo: {end_time - start_time:.1f}s")
        
        if args.show_answer:
            print(f"\n📋 RESPUESTA FINAL:")
            print("-" * 30)
            print(result.final_answer)
        
        print(f"\n💾 Guardado en: {persistence.current_file}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.error(f"Error en ejecución: {e}", exc_info=True)
        return 1

def analyze_command(args):
    """Comando: analizar ejecución guardada"""
    
    jsonl_file = Path(args.file)
    
    if not jsonl_file.exists():
        print(f"❌ Archivo no encontrado: {jsonl_file}")
        return 1
    
    print(f"🔍 Analizando: {jsonl_file}")
    print("=" * 60)
    
    analyzer = RunAnalyzer(jsonl_file)
    analysis = analyzer.analyze()
    
    # Mostrar análisis
    metadata = analysis["run_metadata"]
    print(f"🆔 Run ID: {metadata['run_id']}")
    print(f"❓ Pregunta: {metadata['question'][:80]}...")
    print(f"🤖 Modelos: {', '.join(metadata['models'])}")
    print(f"⏱️ Duración: {metadata['duration']:.1f}s")
    print(f"🎯 Resultado: {metadata['consensus_type']} en {metadata['iterations']} iteraciones")
    print(f"⭐ Calidad final: {metadata['final_quality']:.3f}")
    
    # Progresión de consenso
    consensus = analysis["consensus_analysis"]
    print(f"\n📈 PROGRESIÓN DEL CONSENSO:")
    for evolution in consensus["consensus_evolution"]:
        iter_num = evolution["iteration"]
        votes = evolution["consensus_votes"]
        quality = evolution["average_quality"]
        convergence = evolution["convergence_score"]
        
        print(f"  Iteración {iter_num}: {votes} votos, calidad {quality:.3f}, convergencia {convergence:.3f}")
    
    # Rendimiento de modelos
    performance = analysis["model_performance"]
    print(f"\n🤖 RENDIMIENTO DE MODELOS:")
    for model, stats in performance.items():
        print(f"  {model}: {stats['total_interactions']} interacciones, "
              f"respuesta promedio {stats['avg_response_length']:.0f} chars")
    
    # Mecanismos usados
    mechanisms = analysis["mechanism_usage"]
    if mechanisms["total_mechanisms_used"] > 0:
        print(f"\n🔧 MECANISMOS DE DESBLOQUEO:")
        for mech_type, count in mechanisms["mechanism_breakdown"].items():
            print(f"  {mech_type}: {count} veces")
    
    # Guardar análisis detallado
    if args.save_analysis:
        analysis_file = jsonl_file.with_suffix('.analysis.json')
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Análisis detallado guardado en: {analysis_file}")
    
    return 0

async def benchmark_command(args):
    """Comando: ejecutar benchmark con múltiples preguntas"""
    
    questions_file = Path(args.questions)
    
    if not questions_file.exists():
        print(f"❌ Archivo de preguntas no encontrado: {questions_file}")
        return 1
    
    # Cargar preguntas
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    print(f"🧪 BENCHMARK ACD-MLLM")
    print(f"📝 {len(questions)} preguntas desde {questions_file}")
    print(f"🤖 Modelos: {args.models}")
    print(f"🔄 {args.iterations} iteraciones máximas")
    print("=" * 60)
    
    # Crear modelos
    models = create_models_from_args(args.models, args.config)
    if not models:
        print("❌ No se pudieron configurar modelos.")
        return 1
    
    # Configurar persistencia
    persistence = PersistenceManager(args.output)
    runner = ACDMLLMWithPersistence(persistence)
    
    results = []
    
    for i, question in enumerate(questions):
        print(f"\n📋 Pregunta {i+1}/{len(questions)}")
        print(f"❓ {question[:100]}...")
        
        config = {
            "max_iterations": args.iterations,
            "benchmark_mode": True,
            "question_index": i
        }
        
        start_time = time.time()
        try:
            result = await runner.run_with_logging(question, models, config)
            duration = time.time() - start_time
            
            results.append({
                "question": question,
                "consensus_type": result.consensus_type.value,
                "iterations": result.iterations_taken,
                "quality": result.final_quality_score,
                "confidence": result.confidence_level,
                "duration": duration
            })
            
            print(f"✅ {result.consensus_type.value} en {result.iterations_taken} iter, "
                  f"calidad {result.final_quality_score:.3f}, {duration:.1f}s")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "question": question,
                "error": str(e),
                "duration": time.time() - start_time
            })
    
    # Resumen del benchmark
    print(f"\n🏁 RESUMEN DEL BENCHMARK")
    print("=" * 40)
    
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    print(f"✅ Exitosas: {len(successful)}/{len(questions)}")
    print(f"❌ Fallidas: {len(failed)}")
    
    if successful:
        avg_quality = sum(r["quality"] for r in successful) / len(successful)
        avg_duration = sum(r["duration"] for r in successful) / len(successful)
        avg_iterations = sum(r["iterations"] for r in successful) / len(successful)
        
        consensus_types = {}
        for r in successful:
            consensus_type = r["consensus_type"]
            consensus_types[consensus_type] = consensus_types.get(consensus_type, 0) + 1
        
        print(f"⭐ Calidad promedio: {avg_quality:.3f}")
        print(f"🔄 Iteraciones promedio: {avg_iterations:.1f}")
        print(f"⏱️ Duración promedio: {avg_duration:.1f}s")
        print(f"🎯 Tipos de consenso: {consensus_types}")
    
    # Guardar resultados del benchmark
    benchmark_file = Path(args.output) / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(benchmark_file, 'w', encoding='utf-8') as f:
        json.dump({
            "benchmark_metadata": {
                "questions_file": str(questions_file),
                "models": args.models,
                "max_iterations": args.iterations,
                "timestamp": datetime.now().isoformat()
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: {benchmark_file}")
    
    return 0

def main():
    """Función principal del CLI"""
    
    parser = argparse.ArgumentParser(
        description="ACD-MLLM: Algoritmo de Convergencia Deliberativa Multi-LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  acd-mllm run "¿Cuál es la mejor estrategia de marketing?" --models gpt-4 claude
  acd-mllm analyze runs/run_20240101_123456.jsonl --save-analysis
  acd-mllm benchmark questions.txt --models gpt-4 claude gemini --iterations 5
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Logging verbose')
    parser.add_argument('--log-file', help='Archivo de log opcional')
    parser.add_argument('--output', '-o', default='runs', help='Directorio de salida (default: runs)')
    parser.add_argument('--config', '-c', help='Archivo de configuración JSON')
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Comando: run
    run_parser = subparsers.add_parser('run', help='Ejecutar consenso con una pregunta')
    run_parser.add_argument('question', help='Pregunta para el consenso')
    run_parser.add_argument('--models', '-m', nargs='+', 
                           choices=['gpt-4', 'gpt-3.5', 'claude', 'gemini'],
                           default=['gpt-4', 'claude'],
                           help='Modelos a usar (default: gpt-4 claude)')
    run_parser.add_argument('--max-iterations', type=int, default=10, 
                           help='Número máximo de iteraciones')
    run_parser.add_argument('--quality-threshold', type=float, default=0.85,
                           help='Umbral de calidad para consenso')
    run_parser.add_argument('--similarity-threshold', type=float, default=0.90,
                           help='Umbral de similitud para convergencia')
    run_parser.add_argument('--show-answer', action='store_true',
                           help='Mostrar respuesta final en consola')
    
    # Comando: analyze
    analyze_parser = subparsers.add_parser('analyze', help='Analizar ejecución guardada')
    analyze_parser.add_argument('file', help='Archivo JSONL a analizar')
    analyze_parser.add_argument('--save-analysis', action='store_true',
                               help='Guardar análisis detallado en JSON')
    
    # Comando: benchmark
    benchmark_parser = subparsers.add_parser('benchmark', help='Ejecutar benchmark con múltiples preguntas')
    benchmark_parser.add_argument('questions', help='Archivo de texto con preguntas (una por línea)')
    benchmark_parser.add_argument('--models', '-m', nargs='+',
                                 choices=['gpt-4', 'gpt-3.5', 'claude', 'gemini'],
                                 default=['gpt-4', 'claude'],
                                 help='Modelos a usar')
    benchmark_parser.add_argument('--iterations', type=int, default=5,
                                 help='Iteraciones máximas por pregunta')
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging(args.verbose, args.log_file)
    
    # Crear directorio de salida
    Path(args.output).mkdir(exist_ok=True)
    
    # Ejecutar comando
    if args.command == 'run':
        return asyncio.run(run_consensus_command(args))
    elif args.command == 'analyze':
        return analyze_command(args)
    elif args.command == 'benchmark':
        return asyncio.run(benchmark_command(args))
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
