from src.core.plugins.context import PluginContext


def register(context: PluginContext):
    if context._app:
        from routes.recommend import create_router

        context.include_router(create_router(context))
