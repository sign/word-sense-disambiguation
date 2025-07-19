import os
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

templates = Jinja2Templates(directory=os.path.dirname(__file__))


def _dataclasses_to_dict(obj):
    if hasattr(obj, '__dataclass_fields__'):
        return {field.name: _dataclasses_to_dict(getattr(obj, field.name))
                for field in obj.__dataclass_fields__.values()}
    elif isinstance(obj, list):
        return [_dataclasses_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _dataclasses_to_dict(value) for key, value in obj.items()}
    return obj


async def http_exception_handler(request: Request, exc: Exception):
    status_code = exc.status_code if hasattr(exc, 'status_code') else 500
    body = {
        "error": {
            "status": status_code,
            "message": exc.detail if hasattr(exc, 'detail') else str(exc),
            "type": type(exc).__name__
        }
    }
    return JSONResponse(body, status_code=status_code)


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
        return JSONResponse(_dataclasses_to_dict(result))


async def index_request(request: Request):
    endpoints = {route.path: str(request.url_for(route.name))
                 for route in routes if len(route.param_convertors) == 0}
    return JSONResponse({'endpoints': endpoints})


async def health_check_request(request: Request):
    body = {
        'status': 'healthy',
        'timestamp': datetime.now(tz=UTC).isoformat(),
        'service': 'wn.web',
    }
    return JSONResponse(body, status_code=200)


async def options_request(request: Request):
    return JSONResponse({}, status_code=200)


routes = [
    Route('/', endpoint=index_request),
    Route('/health', endpoint=health_check_request),
    Route('/disambiguate', endpoint=disambiguate_request),
    Route('/{path:path}', endpoint=options_request, methods=['OPTIONS']),
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

app = Starlette(debug=True, routes=routes, middleware=middlewares,
                exception_handlers={
                    HTTPException: http_exception_handler,
                    Exception: http_exception_handler,
                })
