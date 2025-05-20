"""FastAPI REST transport.

For systems that do not speak MCP. Same service, same behaviour, different
wire format. The app holds one shared :class:`~..service.EnrichmentService`
for its lifetime so the cache and rate limiter are actually shared across
requests rather than rebuilt per call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from ..cache import TTLCache
from ..config import get_settings
from ..errors import AllProvidersFailedError, InvalidEmailError
from ..logging import configure_logging, get_logger, set_correlation_id
from ..models import (
    BatchEnrichmentRequest,
    BatchEnrichmentResult,
    EnrichmentRequest,
    EnrichmentResult,
    ParsedEmail,
)
from ..parsing import parse_email
from ..providers.base import available_providers
from ..service import EnrichmentService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the service once at startup and dispose of it at shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    app.state.service = EnrichmentService(settings)
    logger.info(
        "api starting",
        extra={
            "providers": settings.enabled_providers,
            "cache_enabled": settings.cache_enabled,
        },
    )
    try:
        yield
    finally:
        await app.state.service.close()
        logger.info("api stopped")


app = FastAPI(
    title="Email Enrichment API",
    description=(
        "Resolve a person from an email address. Provider-agnostic, with "
        "confidence and provenance on every returned field."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def get_service(request: Request) -> EnrichmentService:
    """Dependency handing the shared service to route handlers."""
    return request.app.state.service


ServiceDep = Annotated[EnrichmentService, Depends(get_service)]


@app.middleware("http")
async def correlation_middleware(request: Request, call_next: Any) -> Any:
    """Attach a correlation ID to every request and echo it back.

    Honours an inbound ``X-Correlation-ID`` so a trace can be followed across
    service boundaries rather than restarting at this hop.
    """
    correlation_id = set_correlation_id(request.headers.get("X-Correlation-ID"))
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


@app.exception_handler(InvalidEmailError)
async def _invalid_email_handler(_: Request, exc: InvalidEmailError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "invalid_email", "detail": str(exc)},
    )


@app.exception_handler(AllProvidersFailedError)
async def _providers_failed_handler(
    _: Request, exc: AllProvidersFailedError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"error": "all_providers_failed", "detail": exc.errors},
    )


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness probe. Intentionally does no work beyond confirming the process."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
async def ready(service: ServiceDep) -> dict[str, Any]:
    """Readiness probe, including cache statistics when available."""
    payload: dict[str, Any] = {
        "status": "ready",
        "providers": [provider.name for provider in service.providers],
        "registered_providers": available_providers(),
    }
    if isinstance(service.cache, TTLCache):
        payload["cache"] = await service.cache.stats()
    return payload


@app.post("/v1/enrich", response_model=EnrichmentResult, tags=["enrichment"])
async def enrich(request: EnrichmentRequest, service: ServiceDep) -> EnrichmentResult:
    """Enrich a single email address."""
    return await service.enrich(request)


@app.post(
    "/v1/enrich/batch", response_model=BatchEnrichmentResult, tags=["enrichment"]
)
async def enrich_batch(
    request: BatchEnrichmentRequest, service: ServiceDep
) -> BatchEnrichmentResult:
    """Enrich up to 100 addresses concurrently."""
    return await service.enrich_batch(
        [str(email) for email in request.emails], skip_cache=request.skip_cache
    )


@app.post("/v1/classify", response_model=ParsedEmail, tags=["enrichment"])
async def classify(
    email: Annotated[str, Body(embed=True, examples=["jane.doe@acme.com"])],
) -> ParsedEmail:
    """Classify an address without enriching it. Free and instant."""
    try:
        return parse_email(email)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


def main() -> None:
    """Console-script entry point."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "enrich.server.api:app",
        host=settings.host,
        port=settings.port,
        log_config=None,  # our own structured logging is already installed
    )


if __name__ == "__main__":
    main()
