def dashboard_view(request):
    try:
        results = perform_fraud_detection()
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
