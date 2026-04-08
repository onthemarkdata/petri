"""petri scan command."""

from __future__ import annotations

import typer

from petri.cli._bootstrap import find_petri_dir


def register(app: typer.Typer) -> None:
    @app.command()
    def scan(
        fix: bool = typer.Option(False, "--fix", help="Auto-fix issues"),
        loop: bool = typer.Option(
            False, "--loop", help="Repeat until clean"
        ),
    ) -> None:
        """Run the contradiction scanner across the current .petri/ project."""
        from petri.analysis.scanner import auto_fix as scanner_auto_fix
        from petri.analysis.scanner import scan as run_scan
        from petri.analysis.scanner import scan_loop

        petri_dir = find_petri_dir()

        # Determine generated config dir (default: .claude/ in project root)
        generated_dir = petri_dir.parent / ".claude"
        if not generated_dir.exists():
            generated_dir = None

        if loop:
            issues = scan_loop(petri_dir, generated_dir)
        elif fix:
            issues = run_scan(petri_dir, generated_dir)
            fixable = [issue for issue in issues if issue.fix_path]
            if fixable:
                fixed = scanner_auto_fix(fixable)
                typer.echo(f"Auto-fixed {len(fixed)} issues.")
            issues = run_scan(petri_dir, generated_dir)
        else:
            issues = run_scan(petri_dir, generated_dir)

        if not issues:
            typer.echo("No inconsistencies found.")
        else:
            typer.echo(f"\nFound {len(issues)} inconsistencies:\n")
            for index, issue in enumerate(issues, 1):
                typer.echo(f"  {index}. [{issue.category}] {issue.description}")
                if issue.file_path:
                    typer.echo(f"     File: {issue.file_path}")

        raise typer.Exit(code=0 if not issues else 1)
