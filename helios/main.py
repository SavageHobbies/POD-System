from __future__ import annotations

import asyncio
from dataclasses import asdict

import typer
from rich import print
from rich.panel import Panel

from .config import HeliosConfig
from .agents.ceo import run_ceo
from .utils.jsonio import dumps
from .mcp_stub import serve as mcp_serve
from .providers.etsy import EtsyClient

app = typer.Typer(add_completion=False)


@app.command()
def run(dry_run: bool = typer.Option(False, help="Run with mock data")) -> None:
    config = HeliosConfig.load()
    try:
        result = asyncio.run(run_ceo(config, dry_run=dry_run))
    except Exception as e:
        print(Panel.fit(f"[red]Execution failed[/red]\n{e}"))
        raise typer.Exit(code=1)

    print(Panel.fit("Helios CEO Result (truncated)"))
    summary = {
        "execution_summary": asdict(result.execution_summary),
        "trend_data": asdict(result.trend_data),
        "audience_insights": asdict(result.audience_insights),
        "product_portfolio": result.product_portfolio,
        "creative_concepts": result.creative_concepts[:1],
        "marketing_materials": result.marketing_materials[:2],
        "publication_queue": result.publication_queue[:2],
    }
    print(dumps(summary))


@app.command()
def demo() -> None:
    print("See README for usage")


@app.command()
def mcp_stub(host: str = "127.0.0.1", port: int = 8787) -> None:
    print(Panel.fit(f"Starting MCP stub on http://{host}:{port}"))
    try:
        mcp_serve(host=host, port=port)
    except KeyboardInterrupt:
        print("Shutting down MCP stub")


def _etsy_from_env(cfg: HeliosConfig) -> EtsyClient:
    if not cfg.etsy_api_key:
        raise typer.BadParameter("ETSY_API_KEY is required in .env")
    if not cfg.etsy_oauth_token:
        raise typer.BadParameter("ETSY_OAUTH_TOKEN is required in .env")
    return EtsyClient(api_key=cfg.etsy_api_key, oauth_token=cfg.etsy_oauth_token)


@app.command()
def etsy_me() -> None:
    cfg = HeliosConfig.load()
    client = _etsy_from_env(cfg)
    data = asyncio.run(client.get_me())
    print(dumps(data))


@app.command()
def etsy_shops() -> None:
    cfg = HeliosConfig.load()
    client = _etsy_from_env(cfg)
    me = asyncio.run(client.get_me())
    user_id = me.get("user_id") or me.get("shop_id") or me.get("id")
    if not user_id:
        raise typer.Exit(code=1)
    data = asyncio.run(client.get_shops_by_user(int(user_id)))
    print(dumps(data))


@app.command()
def etsy_taxonomy(kind: str = typer.Option("buyer", help="buyer|seller")) -> None:
    cfg = HeliosConfig.load()
    client = _etsy_from_env(cfg)
    if kind == "buyer":
        data = asyncio.run(client.get_buyer_taxonomy_nodes())
    else:
        data = asyncio.run(client.get_seller_taxonomy_nodes())
    print(dumps(data))


@app.command()
def etsy_shipping_profiles(shop_id: str = typer.Option("", help="Override shop id")) -> None:
    cfg = HeliosConfig.load()
    client = _etsy_from_env(cfg)
    sid = int(shop_id or (cfg.etsy_shop_id or 0))
    if not sid:
        print("Missing shop_id. Provide --shop-id or set ETSY_SHOP_ID in .env")
        raise typer.Exit(code=1)
    data = asyncio.run(client.get_shipping_profiles(sid))
    print(dumps(data))


@app.command()
def etsy_return_policies(shop_id: str = typer.Option("", help="Override shop id")) -> None:
    cfg = HeliosConfig.load()
    client = _etsy_from_env(cfg)
    sid = int(shop_id or (cfg.etsy_shop_id or 0))
    if not sid:
        print("Missing shop_id. Provide --shop-id or set ETSY_SHOP_ID in .env")
        raise typer.Exit(code=1)
    data = asyncio.run(client.get_return_policies(sid))
    print(dumps(data))

if __name__ == "__main__":
    app()
