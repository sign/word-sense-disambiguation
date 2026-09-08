import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.templating import Jinja2Templates

from wsd.env import WORDNET_URL
from wsd.word_sense_disambiguation import disambiguate

# Honor LOG_LEVEL from the environment so the Dockerfile (or a local operator)
# can dial in verbosity without touching code. Defaulting to INFO means our
# module loggers (e.g. wsd.word_sense_disambiguation) emit their warnings and
# above even though nothing else configures root logging.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

templates = Jinja2Templates(directory=os.path.dirname(__file__))


async def exception_handler(request: Request, exc: Exception):
    status = exc.status_code if isinstance(exc, HTTPException) else 500
    message = exc.detail if isinstance(exc, HTTPException) else str(exc)
    body = {"error": {"status": status, "message": message, "type": type(exc).__name__}}
    return JSONResponse(body, status_code=status)


async def disambiguate_request(request: Request):
    params = request.query_params
    if "text" not in params:
        raise HTTPException(status_code=400, detail="Missing 'text' query parameter")
    if "lang" not in params:
        raise HTTPException(status_code=400, detail="Missing 'lang' query parameter")

    result = disambiguate(text=params["text"], language=params["lang"])

    # Check if HTML output is requested
    if params.get("output") == "html":
        return templates.TemplateResponse("wsd.template.html", {
            "tokens": result.tokens,
            "entities": result.entities,
            "request": request,
            "wordnet_url": WORDNET_URL,
        })
    else:
        return JSONResponse(asdict(result))


async def index_request(request: Request):
    endpoints = {route.path: str(request.url_for(route.name))
                 for route in routes if len(route.param_convertors) == 0}
    return JSONResponse({'endpoints': endpoints})


async def health_check_request(request: Request):
    body = {
        'status': 'healthy',
        'timestamp': datetime.now(tz=UTC).isoformat(),
        'service': 'wsd.server',
    }
    return JSONResponse(body, status_code=200)


routes = [
    Route('/', endpoint=index_request),
    Route('/health', endpoint=health_check_request),
    Route('/disambiguate', endpoint=disambiguate_request),
]

middlewares = [
    Middleware(GZipMiddleware, minimum_size=1000, compresslevel=9),
    Middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_methods=['*'],
        allow_headers=['*'],
    )
]

@asynccontextmanager
async def lifespan(app: Starlette):
    """Load spaCy and the WSD model before serving (Cloud Run routes traffic only once the port
    is open, so this costs start-up time, not request latency). WSD_WARMUP=0 skips it for tests."""
    if os.environ.get("WSD_WARMUP", "1") != "0":
        logging.getLogger(__name__).info("Warming up the pipeline...")
        disambiguate("bank")
    yield


app = Starlette(routes=routes, middleware=middlewares, lifespan=lifespan,
                exception_handlers={HTTPException: exception_handler, Exception: exception_handler})
