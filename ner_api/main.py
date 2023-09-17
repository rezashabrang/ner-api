"""Customizing fast api."""
import tensorflow as tf
import os

tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_inter_op_parallelism_threads(int(os.getenv("TF_PARALLEL_CORES")))
tf.config.threading.set_intra_op_parallelism_threads(int(os.getenv("TF_PARALLEL_CORES")))
from typing import Union


from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.openapi.utils import get_openapi
from ner_api.routers import http_ner
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)

app = FastAPI(docs_url=None)

app.mount(
    "/app/ner_api/static/",
    StaticFiles(directory="/app/ner_api/static/"),
    name="static",
)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/app/ner_api/static/bundle.js",
        swagger_css_url="/app/ner_api/static/swagger.css",
    )
DESCRIPTION = """
"""


async def get_token_header(x_token: str = Header(...)) -> Union[None, Exception]:
    """."""
    if x_token != os.getenv("API_KEY"):
        raise HTTPException(status_code=400, detail="X-Token header invalid")
    return None


def custom_openapi():
    """Defining custom API schema."""
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Named Entity Recognition API",
        version="1.0",
        description=DESCRIPTION,
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore

app.include_router(
    http_ner.router,
    prefix=os.getenv("ROOT_PATH", ""),
    # dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not found"}},
)
