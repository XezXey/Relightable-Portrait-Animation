from flask import Flask, request, send_file, send_from_directory
import glob, os
import numpy as np
import json
import sys
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument('--sampling_dir', default='/data/mint/sampling')
parser.add_argument('--sample_pair_json', required=True)
parser.add_argument('--num_frames', type=int, default=60)
parser.add_argument('--set_', default='valid')
parser.add_argument('--port', required=True)
parser.add_argument('--host', default='0.0.0.0')
args = parser.parse_args()

def sort_by_frame(path_list):
    frame_anno = []
    for p in path_list:
        frame_idx = os.path.splitext(p.split('/')[-1].split('_')[-1])[0][5:]   # 0-4 is "frame", so we used [5:] here
        frame_anno.append(int(frame_idx))
    sorted_idx = np.argsort(frame_anno)
    sorted_path_list = []
    for idx in sorted_idx:
      sorted_path_list.append(path_list[idx])
    return sorted_path_list

def create_app():
    app = Flask(__name__)
    
    @app.route('/files/<path:path>')
    def servefile(path):
        #NOTE: Serve the file to html    
        return send_from_directory('/', path)
    
    @app.route('/')
    def root():
        # Fixed the training step and varying the diffusion step
        out = """<style>
                th, tr, td{
                    border:1px solid black;margin-left:auto;margin-right:auto;text-align: center;
                }
                </style>"""
        
        out += "<script>"
        out += """
        function transposeTable(table) {
            var transposedTable = document.createElement("table");

            for (var i = 0; i < table.rows[0].cells.length; i++) {
                var newRow = transposedTable.insertRow(i);

                for (var j = 0; j < table.rows.length; j++) {
                var newCell = newRow.insertCell(j);
                newCell.innerHTML = table.rows[j].cells[i].innerHTML;
                }
            }

            table.parentNode.replaceChild(transposedTable, table);
        }

        function transposeAllTables() {
            var tables = document.getElementsByTagName("table");

            for (var i = 0; i < tables.length; i++) {
                transposeTable(tables[i]);
            }
        }

        """
        out += "</script>"
        
        s = request.args.get('s', 0)
        e = request.args.get('e', 1)
        show_frame = request.args.get('show_frame', "0").split(",")
        sample_json = str(request.args.get('sample_json', args.sample_pair_json))
        
        data_path = f"/data/mint/DPM_Dataset/ffhq_256_with_anno/ffhq_256_no_aliasing_png/{args.set_}/"
        try:
            os.path.isfile(sample_json)
            f = open(sample_json)
            sample_pairs = json.load(f)['pair']
        except:
            raise ValueError(f"Sample json file not found: {sample_json}")

        out += f"<h2> Sample json file: {sample_json} {args.num_frames} </h2>"

        count = 0
        to_show = list(sample_pairs.items())[int(s):int(e)]
        
        # Showing videos       
        for ts in to_show:
            count += 1
            out += "<table>"
            pid, v = ts
            src = v['src']
            dst = v['dst']
            
            out += f"[#{pid}] {src}=>{dst} : <img src=/files/{data_path}/{src.replace('jpg', 'png')}>, {dst} : <img src=/files/{data_path}/{dst.replace('jpg', 'png')}>" + "<br>" + "<br>"
            # Add col label
            out += f"<tr><td>"
            for gs in [2.0, 4.0, 4.5, 6.0, 8.0]:
                out += f"<td style='font-weight:bold;font-size:25pt'>gs={gs}</td>"
                
            for scale_sh in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                out += f"<tr>"
                out += f"<td style='font-weight:bold;font-size:25pt'>scale_sh={scale_sh}</td>"
                for gs in [2.0, 4.0, 4.5, 6.0, 8.0]:
                    img_dir = f'{args.sampling_dir}/src={src}_dst={dst}/scale_sh={scale_sh}/gs={gs}_ds=25/n_step={args.num_frames}/'
                    # vid = f'{img_dir}/out_rt.mp4'
                    vid = f'{img_dir}/res_rt.mp4'
                    out += f"<td>"
                    if os.path.exists(vid):
                        out += "<video width=\"256\" height=\"256\" muted autoplay loop><source src=/files/" + vid + " type=\"video/mp4\"></video>"
                    else:
                        out += "<p style=\"color:red\">File not found!</p>"
                    out += "</td>"
                    ###################################################
                    
                out += "</tr>"
                    
            out += "</table>"
            out += "<br> <hr>"
        
        # Showing images
        for sf in show_frame:
            if int(sf) > args.num_frames: continue
            for ts in to_show:
                count += 1
                out += "<table>"
                pid, v = ts
                src = v['src']
                dst = v['dst']
                
                out += f"<p style='font-weight:bold;font-size:25pt'>Frame={sf}</p>"
                out += f"<tr><td>"
                for gs in [2.0, 4.0, 4.5, 6.0, 8.0]:
                    out += f"<td style='font-weight:bold;font-size:25pt'>gs={gs}</td>"
                # Model 
                for scale_sh in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    out += "<tr>"
                    out += f"<td style='font-weight:bold;font-size:25pt'>scale_sh={scale_sh}</td>"
                    for gs in [2.0, 4.0, 4.5, 6.0, 8.0]:
                        img_dir = f'{args.sampling_dir}/src={src}_dst={dst}/scale_sh={scale_sh}/gs={gs}_ds=25/n_step={args.num_frames}/'
                        frames = sorted(glob.glob(f'{img_dir}/res_frame*.png'))
                        # Model's metadata
                        out += f"<td>"
                        if len(frames) > 1:
                            f = frames[int(sf)]
                            out += "<img width=\"256\" height=\"256\" src=/files/" + f + ">"
                        else:
                            out += "<p style=\"color:red\">File not found!</p>"
                        out += "</td>"
                        ###################################################
                        
                    out += "</tr>"
            
            out += "</table>"
            out += "<br> <hr>"
        
                    
        return out

    return app

if __name__ == "__main__":
    
    app = create_app()
    app.run(host=args.host, port=args.port, debug=True, threaded=True)
