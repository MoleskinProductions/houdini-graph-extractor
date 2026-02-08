"""CLI entry point for Houdini Graph Extractor."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .config import Config, get_config, set_config
from .ingestion.youtube import YouTubeIngester
from .ingestion.frame_extractor import FrameExtractor
from .analysis.transcript_parser import TranscriptParser
from .analysis.visual_extractor import VisualExtractor
from .analysis.llm_transcript_analyzer import LLMTranscriptAnalyzer
from .analysis.harmonizer import Harmonizer, HarmonizerConfig
from .state.graph_state import GraphStateManager
from .state.merger import StateMerger, MergeConfig
from .output.hou_data import HouDataFormatter

console = Console()


@click.command()
@click.argument("url")
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="output.json",
    help="Output JSON file path",
)
@click.option(
    "--resolution",
    type=str,
    default="1080",
    help="Video resolution to download (default: 1080)",
)
@click.option(
    "--api-base",
    type=str,
    default="https://generativelanguage.googleapis.com/v1beta/openai/",
    help="VLM API base URL (Google, OpenRouter, vLLM, Ollama)",
)
@click.option(
    "--model",
    type=str,
    default="gemini-2.5-flash",
    help="Vision-language model name",
)
@click.option(
    "--api-key",
    type=str,
    envvar="GOOGLE_API_KEY",
    help="API key (or set GOOGLE_API_KEY env var)",
)
@click.option(
    "--temp-dir",
    type=click.Path(),
    default="/tmp/houdini-extractor",
    help="Temporary directory for downloaded files",
)
@click.option(
    "--sample-interval",
    type=float,
    default=5.0,
    help="Frame sampling interval in seconds",
)
@click.option(
    "--buffer-after",
    type=float,
    default=1.0,
    help="Seconds to sample after action events",
)
@click.option(
    "--skip-low-priority",
    is_flag=True,
    help="Skip low priority frames during extraction",
)
@click.option(
    "--use-llm-transcript/--no-llm-transcript",
    default=True,
    help="Use LLM for transcript analysis (default: enabled)",
)
@click.option(
    "--harmonize/--no-harmonization",
    default=True,
    help="Enable transcript/visual harmonization (default: enabled)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def main(
    url: str,
    output: str,
    resolution: str,
    api_base: str,
    model: str,
    api_key: str,
    temp_dir: str,
    sample_interval: float,
    buffer_after: float,
    skip_low_priority: bool,
    use_llm_transcript: bool,
    harmonize: bool,
    verbose: bool,
):
    """
    Extract Houdini node graphs from YouTube tutorial videos.

    URL: YouTube video URL to process

    This tool uses an incremental harmonization pipeline that synchronizes
    transcript analysis with visual frame extraction for improved accuracy.
    """
    # Configure
    config = Config()
    config.temp_dir = Path(temp_dir)
    config.vlm.api_base = api_base
    config.vlm.model_name = model
    config.extraction.frame_interval_seconds = sample_interval
    config.transcript_analysis.use_llm = use_llm_transcript
    config.harmonization.enabled = harmonize
    set_config(config)

    output_path = Path(output)

    console.print(f"[bold blue]Houdini Graph Extractor[/bold blue]")
    console.print(f"Processing: {url}")
    console.print(f"[dim]Mode: {'LLM' if use_llm_transcript else 'Regex'} transcript analysis, "
                  f"{'Harmonization enabled' if harmonize else 'No harmonization'}[/dim]\n")

    try:
        # Step 1: Ingest video and transcript
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading video and transcript...", total=None)

            ingester = YouTubeIngester(config.temp_dir)
            video_info = ingester.ingest(url, resolution)

            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Video: {video_info.title}")
        console.print(f"[green]✓[/green] Duration: {video_info.duration_seconds:.0f}s")
        console.print(f"[green]✓[/green] Transcript segments: {len(video_info.transcript)}")

        # Step 2: Analyze transcript
        console.print("\n[bold]Analyzing transcript...[/bold]")

        if use_llm_transcript and video_info.transcript:
            console.print(f"[dim]Using LLM-based entity extraction[/dim]")
            llm_analyzer = LLMTranscriptAnalyzer(api_base, model, api_key)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("LLM analyzing transcript...", total=None)
                enhanced_events = llm_analyzer.analyze(
                    video_info.transcript,
                    use_fallback_on_error=config.transcript_analysis.fallback_on_error,
                )
                progress.update(task, completed=True)

            action_events = llm_analyzer.get_action_events(enhanced_events)
            console.print(f"[green]✓[/green] Enhanced action events: {len(action_events)}")

            # Generate timestamps from enhanced events
            timestamps = llm_analyzer.to_sampling_timestamps(
                enhanced_events,
                buffer_after=buffer_after,
                interval=sample_interval,
            )
        else:
            # Fall back to regex-based parsing
            console.print(f"[dim]Using regex-based parsing[/dim]")
            parser = TranscriptParser()
            regex_events = parser.parse(video_info.transcript)
            action_events = regex_events  # For timeline output
            enhanced_events = []  # No enhanced events in regex mode

            console.print(f"[green]✓[/green] Action events detected: {len(regex_events)}")

            # Generate timestamps from regex events
            if regex_events:
                timestamps = parser.get_sampling_timestamps(
                    regex_events,
                    buffer_after=buffer_after,
                    interval=sample_interval,
                )
            else:
                timestamps = []

        if verbose and action_events:
            console.print("\n[dim]Detected events:[/dim]")
            for event in action_events[:10]:
                if hasattr(event, 'event_type'):
                    event_type = event.event_type.value if hasattr(event.event_type, 'value') else event.event_type
                    raw_text = event.raw_text[:50] if hasattr(event, 'raw_text') else str(event)[:50]
                    console.print(f"  [{event.timestamp:.1f}s] {event_type}: {raw_text}...")
            if len(action_events) > 10:
                console.print(f"  [dim]... and {len(action_events) - 10} more[/dim]")

        # Step 3: Generate sampling timestamps if none from events
        if not timestamps:
            console.print("[yellow]No action events found, using interval sampling[/yellow]")
            timestamps = [
                {"time": t, "priority": "normal", "source_event": None}
                for t in range(0, int(video_info.duration_seconds), int(sample_interval * 10))
            ]

        console.print(f"[green]✓[/green] Sampling {len(timestamps)} frames")

        # Step 4: Extract frames
        console.print("\n[bold]Extracting frames...[/bold]")
        frame_extractor = FrameExtractor(config.temp_dir)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Extracting {len(timestamps)} frames...", total=None)
            frames = frame_extractor.extract_frames_at_timestamps(
                video_info.video_path,
                timestamps,
            )
            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Extracted {len(frames)} frames")

        # Step 5: Initialize state and harmonizer
        console.print("\n[bold]Processing frames with incremental harmonization...[/bold]")
        console.print(f"[dim]Using model: {model}[/dim]")

        state = GraphStateManager()
        visual_extractor = VisualExtractor(api_base, model, api_key)

        if harmonize:
            harmonizer_config = HarmonizerConfig(
                transcript_window_before=config.harmonization.transcript_window_before,
                transcript_window_after=config.harmonization.transcript_window_after,
                prefer_visual_on_mismatch=config.harmonization.prefer_visual_on_mismatch,
                accept_extra_visual_nodes=config.harmonization.accept_extra_visual_nodes,
                extra_visual_penalty=config.harmonization.extra_visual_penalty,
            )
            harmonizer = Harmonizer(state, harmonizer_config)
        else:
            merger_config = MergeConfig(
                transcript_boost=config.state.transcript_boost,
                decay_rate=config.state.confidence_decay_rate,
                decay_after_frames=config.state.confidence_decay_after_frames,
            )
            merger = StateMerger(state, merger_config)

        # Step 6: Incremental processing loop
        valid_extractions = 0
        total_conflicts = 0
        total_corroborations = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing frames...", total=len(frames))

            for i, frame in enumerate(frames):
                if skip_low_priority and frame.priority == "low":
                    progress.advance(task)
                    continue

                try:
                    # Extract visual topology
                    extraction = visual_extractor.extract_topology(frame.path, frame.timestamp)

                    if extraction.graph_visible and extraction.nodes:
                        valid_extractions += 1

                        if harmonize and enhanced_events:
                            # Harmonize with transcript events
                            result = harmonizer.harmonize(extraction, enhanced_events)
                            total_conflicts += len(result.conflicts)
                            total_corroborations += result.corroborations

                            if verbose:
                                console.print(
                                    f"[dim]  Frame {i} ({frame.timestamp:.1f}s): "
                                    f"{len(extraction.nodes)} nodes, "
                                    f"{result.corroborations} corroborations, "
                                    f"{len(result.conflicts)} conflicts[/dim]"
                                )
                        else:
                            # Direct merge without harmonization
                            nodes_data = [
                                {
                                    "name": n.name,
                                    "type": n.type,
                                    "position": n.position,
                                    "flags": n.flags,
                                }
                                for n in extraction.nodes
                            ]
                            connections_data = [
                                {
                                    "from_node": c.from_node,
                                    "from_output": c.from_output,
                                    "to_node": c.to_node,
                                    "to_input": c.to_input,
                                }
                                for c in extraction.connections
                            ]
                            state.add_visual_extraction(
                                nodes=nodes_data,
                                connections=connections_data,
                                timestamp=extraction.timestamp,
                                extraction_confidence=extraction.extraction_confidence,
                                network_context=extraction.network_context,
                                parent_path=extraction.parent_path,
                            )

                            if verbose:
                                console.print(
                                    f"[dim]  Frame {i} ({frame.timestamp:.1f}s): "
                                    f"{len(extraction.nodes)} nodes, "
                                    f"{len(extraction.connections)} connections[/dim]"
                                )

                except Exception as e:
                    if verbose:
                        console.print(f"[yellow]  Frame {i}: Error - {e}[/yellow]")

                progress.advance(task)

        console.print(f"[green]✓[/green] Valid extractions: {valid_extractions}/{len(frames)}")

        if harmonize:
            console.print(f"[green]✓[/green] Corroborations: {total_corroborations}")
            if total_conflicts > 0:
                console.print(f"[yellow]![/yellow] Conflicts detected: {total_conflicts}")

        # Step 7: Generate output from state
        console.print("\n[bold]Generating output...[/bold]")

        # Get state data
        state_data = state.to_hou_data(include_metadata=config.output.include_metadata)

        # Build final output
        output_data = {
            "version": "1.0",
            "source": {
                "video_url": url,
                "video_title": video_info.title,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            },
            "networks": state_data["networks"],
            "extraction_metadata": {
                "total_frames_analyzed": len(frames),
                "valid_extractions": valid_extractions,
                "action_events_detected": len(action_events),
                "confidence_score": state.get_average_confidence(),
                "flagged_for_review": [],
                "processing_mode": {
                    "transcript_analyzer": "llm" if use_llm_transcript else "regex",
                    "harmonization": harmonize,
                },
            },
        }

        # Add timeline
        if config.output.include_timeline and action_events:
            timeline = []
            for event in action_events:
                event_type = event.event_type.value if hasattr(event.event_type, 'value') else event.event_type
                raw_text = event.raw_text if hasattr(event, 'raw_text') else str(event)
                timeline.append({
                    "time": event.timestamp,
                    "action": event_type,
                    "raw_text": raw_text,
                })
            output_data["extraction_metadata"]["timeline"] = timeline

        # Add harmonization metadata
        if harmonize and config.output.include_metadata:
            stats = state.get_stats()
            output_data["extraction_metadata"]["harmonization"] = {
                "corroborated_nodes": stats["corroborated_nodes"],
                "conflicts_detected": stats["conflicts_detected"],
                "harmony_score": harmonizer.get_harmony_score() if harmonize else None,
                "params_extracted": stats.get("total_params_extracted", 0),
                "nodes_with_params": stats.get("nodes_with_params", 0),
            }

            # Add confidence metadata per node
            if "_connection_metadata" in state_data:
                output_data["_connection_metadata"] = state_data["_connection_metadata"]

            # Add conflicts
            if "_conflicts" in state_data:
                output_data["extraction_metadata"]["flagged_for_review"] = [
                    {
                        "type": c["conflict_type"],
                        "timestamp": c["timestamp"],
                        "resolution": c.get("resolution"),
                    }
                    for c in state_data["_conflicts"]
                ]

        # Save output
        import json
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        console.print(f"[green]✓[/green] Output saved to: {output_path}")

        # Summary
        console.print("\n[bold green]Extraction Complete![/bold green]")

        if output_data["networks"]:
            network = output_data["networks"][0]
            console.print(f"  Nodes: {len(network['nodes'])}")
            console.print(f"  Connections: {len(network['connections'])}")
            console.print(
                f"  Confidence: {output_data['extraction_metadata']['confidence_score']:.0%}"
            )

            if harmonize:
                harm_meta = output_data["extraction_metadata"].get("harmonization", {})
                console.print(f"  Corroborated: {harm_meta.get('corroborated_nodes', 0)} nodes")
                if harm_meta.get("harmony_score"):
                    console.print(f"  Harmony Score: {harm_meta['harmony_score']:.0%}")
                params_extracted = harm_meta.get("params_extracted", 0)
                if params_extracted > 0:
                    console.print(f"  Parameters: {params_extracted} extracted from transcript")

            flagged = output_data["extraction_metadata"].get("flagged_for_review", [])
            if flagged:
                console.print(f"  [yellow]Flagged for review: {len(flagged)}[/yellow]")
        else:
            console.print("[yellow]  No nodes extracted[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[red]Aborted by user[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@click.command("help-docs")
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="help_corpus.json",
    help="Output JSON file path",
)
@click.option(
    "--zip-path",
    type=click.Path(exists=True),
    default="/opt/hfs21.0/houdini/help/nodes.zip",
    help="Path to Houdini help nodes.zip",
)
@click.option(
    "--contexts",
    type=str,
    default=None,
    help="Comma-separated list of contexts to parse (e.g. sop,dop,vop). Default: all.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def help_docs(output: str, zip_path: str, contexts: str | None, verbose: bool):
    """Parse Houdini help documentation into structured JSON.

    Reads node help files from the Houdini nodes.zip archive and produces
    a JSON corpus keyed by context/internal_name.
    """
    from pathlib import Path
    from .ingestion.help_docs import HelpCorpusParser

    output_path = Path(output)

    console.print("[bold blue]Houdini Help Documentation Parser[/bold blue]")
    console.print(f"Source: {zip_path}")

    ctx_set = None
    if contexts:
        ctx_set = {c.strip() for c in contexts.split(",") if c.strip()}
        console.print(f"Contexts: {', '.join(sorted(ctx_set))}")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Parsing help documentation...", total=None)
            parser = HelpCorpusParser(zip_path=zip_path, contexts=ctx_set)
            corpus = parser.parse()
            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Parsed {corpus.node_count} node docs")
        console.print(f"[green]✓[/green] Contexts: {', '.join(corpus.contexts())}")

        if verbose:
            for ctx in corpus.contexts():
                count = len(corpus.get_by_context(ctx))
                console.print(f"  {ctx}: {count}")

        corpus.save_json(output_path)
        console.print(f"[green]✓[/green] Saved to: {output_path}")

    except FileNotFoundError:
        console.print(f"[red]Error: Zip file not found: {zip_path}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@click.command("node-schema")
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="node_schema.json",
    help="Output JSON file path",
)
@click.option(
    "--hython-path",
    type=click.Path(),
    default="/opt/hfs21.0/bin/hython",
    help="Path to hython executable",
)
@click.option(
    "--categories",
    type=str,
    default=None,
    help="Comma-separated categories to extract (e.g. Sop,Dop,Vop). Default: all.",
)
@click.option(
    "--no-ports",
    is_flag=True,
    help="Skip port extraction (faster, no node instantiation needed)",
)
@click.option(
    "--timeout",
    type=int,
    default=120,
    help="Timeout in seconds for hython subprocess (default: 120)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def node_schema(output: str, hython_path: str, categories: str | None, no_ports: bool,
                timeout: int, verbose: bool):
    """Extract Houdini node type schemas via hython introspection.

    Runs a hython subprocess to extract parameter templates, port definitions,
    and metadata for every node type in Houdini.
    """
    from .ingestion.node_schema import NodeSchemaExtractor

    output_path = Path(output)

    console.print("[bold blue]Houdini Node Schema Extractor[/bold blue]")
    console.print(f"hython: {hython_path}")

    cat_list = None
    if categories:
        cat_list = [c.strip() for c in categories.split(",") if c.strip()]
        console.print(f"Categories: {', '.join(cat_list)}")
    else:
        console.print("Categories: all")

    if no_ports:
        console.print("[dim]Port extraction: disabled[/dim]")

    try:
        extractor = NodeSchemaExtractor(
            hython_path=hython_path,
            categories=cat_list,
            timeout=timeout,
            extract_ports=not no_ports,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting node schemas via hython...", total=None)
            corpus = extractor.extract()
            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Extracted {corpus.node_count} node types")
        console.print(f"[green]✓[/green] Houdini version: {corpus.houdini_version}")
        console.print(f"[green]✓[/green] Contexts: {', '.join(corpus.contexts())}")

        if verbose:
            for ctx in corpus.contexts():
                count = len(corpus.get_by_context(ctx))
                console.print(f"  {ctx}: {count}")

        corpus.save_json(output_path)
        console.print(f"[green]✓[/green] Saved to: {output_path}")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except TimeoutError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@click.command("labs-hda")
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="labs_graphs.json",
    help="Output JSON file path",
)
@click.option(
    "--hython-path",
    type=click.Path(),
    default="/opt/hfs21.0/bin/hython",
    help="Path to hython executable",
)
@click.option(
    "--categories",
    type=str,
    default=None,
    help="Comma-separated categories to extract (e.g. Sop,Dop,Top). Default: all.",
)
@click.option(
    "--library-filter",
    type=str,
    default="SideFXLabs",
    help="Filter HDAs by library path containing this string (default: SideFXLabs)",
)
@click.option(
    "--timeout",
    type=int,
    default=300,
    help="Timeout in seconds for hython subprocess (default: 300)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def labs_hda(output: str, hython_path: str, categories: str | None,
             library_filter: str, timeout: int, verbose: bool):
    """Extract Labs HDA internal graphs via hython introspection.

    Runs a hython subprocess to unlock each Labs HDA, extract its internal
    node network (children, connections, non-default parameters), and produce
    a JSON corpus of HDA graphs.
    """
    from .ingestion.labs_hda import LabsHDAExtractor

    output_path = Path(output)

    console.print("[bold blue]Houdini Labs HDA Graph Extractor[/bold blue]")
    console.print(f"hython: {hython_path}")
    console.print(f"Library filter: {library_filter}")

    cat_list = None
    if categories:
        cat_list = [c.strip() for c in categories.split(",") if c.strip()]
        console.print(f"Categories: {', '.join(cat_list)}")
    else:
        console.print("Categories: all")

    try:
        extractor = LabsHDAExtractor(
            hython_path=hython_path,
            categories=cat_list,
            library_filter=library_filter,
            timeout=timeout,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting Labs HDA graphs via hython...", total=None)
            corpus = extractor.extract()
            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Extracted {corpus.hda_count} Labs HDA graphs")
        console.print(f"[green]✓[/green] Houdini version: {corpus.houdini_version}")
        console.print(f"[green]✓[/green] Contexts: {', '.join(corpus.contexts())}")

        if verbose:
            for ctx in corpus.contexts():
                count = len(corpus.get_by_context(ctx))
                console.print(f"  {ctx}: {count}")

            # Show some stats
            total_nodes = sum(g.node_count for g in corpus.graphs.values())
            total_conns = sum(g.connection_count for g in corpus.graphs.values())
            console.print(f"  Total internal nodes: {total_nodes}")
            console.print(f"  Total internal connections: {total_conns}")

        corpus.save_json(output_path)
        console.print(f"[green]✓[/green] Saved to: {output_path}")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except TimeoutError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@click.command("pattern-mine")
@click.option(
    "--corpus",
    type=click.Path(exists=True),
    required=True,
    help="Path to Labs HDA graph corpus JSON (from houdini-labs-extract)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="patterns.json",
    help="Output JSON file path",
)
@click.option(
    "--schema",
    type=click.Path(exists=True),
    default=None,
    help="Optional node schema JSON (from houdini-schema-extract) for port name enrichment",
)
@click.option(
    "--min-count",
    type=int,
    default=1,
    help="Minimum frequency for a pattern to be included (default: 1)",
)
@click.option(
    "--max-chain-length",
    type=int,
    default=3,
    help="Maximum chain length to mine (default: 3)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def pattern_mine(corpus: str, output: str, schema: str | None, min_count: int,
                 max_chain_length: int, verbose: bool):
    """Mine connection patterns from Labs HDA graph corpus.

    Analyzes the HDA graph corpus to discover connection patterns,
    node co-occurrences, port usage statistics, and chain patterns.
    Produces a PatternCorpus JSON for downstream use by the MCP Server.
    """
    from .ingestion.labs_hda.models import HDAGraphCorpus
    from .analysis.pattern_mining import PatternMiner, SchemaEnricher

    output_path = Path(output)

    console.print("[bold blue]Houdini Connection Pattern Miner[/bold blue]")
    console.print(f"Corpus: {corpus}")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading HDA graph corpus...", total=None)
            hda_corpus = HDAGraphCorpus.load_json(corpus)
            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Loaded {hda_corpus.hda_count} HDA graphs")

        miner = PatternMiner(
            min_pattern_count=min_count,
            max_chain_length=max_chain_length,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Mining connection patterns...", total=None)
            result = miner.mine(hda_corpus)
            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Connection patterns: {result.pattern_count}")
        console.print(f"[green]✓[/green] Node suggestions: {result.suggestion_count}")
        console.print(f"[green]✓[/green] Co-occurrences: {len(result.cooccurrences)}")
        console.print(f"[green]✓[/green] Port usage entries: {len(result.port_usage)}")
        console.print(f"[green]✓[/green] 2-chains: {len(result.chain_patterns_2)}")
        console.print(f"[green]✓[/green] 3-chains: {len(result.chain_patterns_3)}")

        # Optional schema enrichment
        if schema:
            from .ingestion.node_schema.models import SchemaCorpus

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Enriching with port names from schema...", total=None)
                schema_corpus = SchemaCorpus.load_json(schema)
                enricher = SchemaEnricher(schema_corpus)
                enricher.enrich(result)
                progress.update(task, completed=True)

            console.print(f"[green]✓[/green] Enriched with schema ({schema_corpus.node_count} node types)")

        if verbose:
            # Show top patterns
            top_patterns = sorted(
                result.connection_patterns.values(),
                key=lambda p: p.count, reverse=True,
            )[:10]
            if top_patterns:
                console.print("\n[bold]Top connection patterns:[/bold]")
                for p in top_patterns:
                    label = p.edge_key
                    if p.source_output_name or p.dest_input_name:
                        label += f" ({p.source_output_name or '?'} -> {p.dest_input_name or '?'})"
                    console.print(f"  {p.count:4d}x  {label}")

        result.save_json(output_path)
        console.print(f"\n[green]✓[/green] Saved to: {output_path}")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
