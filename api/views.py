from django.shortcuts import render
from .services import perform_fraud_detection

def dashboard_view(request):
    try:
        # Get n_clusters from request, default to 6
        n_clusters_str = request.GET.get('n_clusters', '6')
        try:
            n_clusters = int(n_clusters_str)
            if not (2 <= n_clusters <= 20): # Validate range
                n_clusters = 6
        except ValueError:
            n_clusters = 6

        results = perform_fraud_detection(n_clusters=n_clusters)
        context = {
            'graph_main': results['main'],
            'graph_dist': results['distribution'],
            'graph_amount': results['amount'],
            'n_clusters': results['n_clusters'],
            'inertia': f"{results['inertia']:.2f}",
            'n_iter': results['n_iter'],
            'centroids': results['centroids'],
            'predicted_labels_sample': results['predicted_labels_sample'],
            'stats_table': results['stats_table']
        }
    except Exception as e:
        context = {
            'error': str(e)
        }
    return render(request, 'dashboard.html', context)
