from django.conf import settings
import environ

env = environ.Env()

def analytics_tag(request):
    return {
        'analytics_tag': env.str("ANALYTICS_TAG")
    }

def analytics_link(request):
    return {
        'analytics_link': env.str("ANALYTICS_LINK")
    }