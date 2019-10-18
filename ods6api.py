from flask import Flask, request, jsonify, make_response, send_file, render_template
import pandas as pd
import ods6app as ap
import os



app = Flask(__name__ , static_url_path='/static')

@app.route('/' , methods=['GET'])
def root():
  return app.send_static_file('index.html')

@app.route('/api/v1/municipio/data' , methods=['GET'])
def municipio_data():
  mun = request.args.get('mun' , type=str)
  mun = mun[0]
  ano = request.args.get('ano',type=int)
  ano = ano[0]
  col = request.args.get('col', type=str)
  col = col[0]

  response = {col : ap.get_data(mun, col, ano)}
 
  return jsonify(response)

@app.route('/api/v1/municipio/hist' , methods=['GET'])
def get_history_plot():
  mun = request.args.get('mun' , type=str)
  col = request.args.get('col' , type=str)

  
  bytes_obj = ap.get_history(mun,col)
  return send_file(bytes_obj , attachment_filename='{}_{}.png'.format(mun,col) , mimetype='image/png')

@app.route('/api/v1/estado/map/quali' ,methods=['GET'])
def get_map_plot_quali():
  ano = request.args.get('ano' , type=int)
  col = request.args.get('col' , type=str)

  bytes_obj = ap.plot_qualitative_map(ano,col)
  return send_file(bytes_obj , attachment_filename='{}_{}.png'.format(ano,col) , mimetype='image/png')

@app.route('/api/v1/estado/map/quant' ,methods=['GET'])
def get_map_plot_quant():
  ano = request.args.get('ano' , type=int)
  col = request.args.get('col' , type=str)

  bytes_obj = ap.plot_numerical_map(ano,col)
  return send_file(bytes_obj , attachment_filename='{}_{}.png'.format(ano,col) , mimetype='image/png')

@app.route('/api/v1/estado/hist' , methods=['GET'])
def get_mean_history_plot():
  col = request.args.get('col' , type=str)
  bytes_obj = ap.get_mean_history(col)
  return send_file(bytes_obj , attachment_filename='{}_estado_hist.png'.format(col) , mimetype='image/png')

@app.route('/api/v1/estado/mean', methods=['GET'])
def get_mean_sp():
  ano = request.args.get('ano' , type=int)
  col = request.args.get('col' , type=str)
  response = {'Média_{}'.format(col) : ap.get_mean(col,ano)}
  return jsonify(response)

@app.route('/api/v1/cluster/stats' , methods=['GET'])
def get_cluster_stats():
  cols = request.args.get('cols')
  cols = cols.split(' ')
  ano  = request.args.get('ano' ,type=int)
  n = request.args.get('n', type=int)

  result = ap.cluster(ano,cols, n)
  dict_stats = result[0].groupby('labels')[cols].describe().to_json()

  response = {
    'stats' : dict_stats,
    'silhueta' : result[1],
    'Davies Bouldin' : result[2]
  }

  return jsonify(response)

@app.route('/api/v1/cluster/plots', methods=['GET'])
def get_cluster_plot():
  cols = request.args.get('cols')
  cols = cols.split(' ')
  ano  = request.args.get('ano' ,type=int)
  n = request.args.get('n' ,type=int)

  bytes_obj = ap.plot_cluster(ano,cols,n)
  return send_file(bytes_obj , attachment_filename='Clusterz_{}_{}.png'.format(ano,cols) , mimetype='image/png')


@app.route('/api/v1/cluster/map' , methods=['GET'])
def get_cluster_map():
  cols = request.args.get('cols')
  cols = cols.split(' ')
  ano  = request.args.get('ano' ,type=int)
  n = request.args.get('n' ,type=int)

  bytes_obj = ap.plot_cluster_map(ano,cols,n)
  return send_file(bytes_obj , attachment_filename='Clusterz_{}_{}.png'.format(ano,cols) , mimetype='image/png')

if __name__ == "__main__":
  port = int(os.environ.get("PORT",5000))
  app.run(host='0.0.0.0', port=port)