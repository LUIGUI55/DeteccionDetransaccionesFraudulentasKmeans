from django.shortcuts import render
from .services import perform_fraud_detection

def dashboard_view(request):
    try:
        graph, n_clusters = perform_fraud_detection()
        context = {
            'graph': graph,
            'n_clusters': n_clusters
        }
    except Exception as e:
        context = {
            'error': str(e)
        }
    return render(request, 'dashboard.html', context)
