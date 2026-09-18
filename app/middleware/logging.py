import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logger import get_logger

logger = get_logger("app.middleware.logging")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP requests and responses with timing, status codes, and correlation IDs."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract existing X-Request-ID or generate a new UUID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        # Extract client details
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else ""
        full_path = f"{path}?{query_params}" if query_params else path

        start_time = time.perf_counter()

        logger.debug(
            f"Incoming request: {method} {full_path} from {client_ip} [request_id={request_id}]",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "client_ip": client_ip,
            },
        )

        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            # Attach correlation ID and processing duration to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = str(duration_ms)

            status_code = response.status_code
            log_message = (
                f"HTTP {method} {path} - {status_code} ({duration_ms}ms) [request_id={request_id}]"
            )

            extra_data = {
                "request_id": request_id,
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "client_ip": client_ip,
            }

            if status_code >= 500:
                logger.error(log_message, extra=extra_data)
            elif status_code >= 400:
                logger.warning(log_message, extra=extra_data)
            else:
                logger.info(log_message, extra=extra_data)

            return response

        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.exception(
                f"HTTP {method} {path} failed with unhandled exception after {duration_ms}ms [request_id={request_id}]: {exc}",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "duration_ms": duration_ms,
                    "client_ip": client_ip,
                },
            )
            raise exc
