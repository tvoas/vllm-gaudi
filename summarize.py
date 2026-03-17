import csv
import sys
import os
from collections import defaultdict

def summarize_compile_times(file_path, cards):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Dictionary to hold aggregated logic: stats[phase] = { ... }
    stats = defaultdict(lambda: {'count': 0, 'total_duration': 0.0, 'total_comps': 0, 'max_duration': 0.0, 'max_comps': 0})
    
    total_time = 0.0
    total_comps = 0

    outq = []
    
    with open(file_path, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = row.get('phase', 'UNKNOWN')
            
            try:
                duration = float(row.get('duration_sec', 0))
                comps = int(row.get('compilations', 0))
            except ValueError:
                continue # Skip slightly malformed rows
                
            stats[phase]['count'] += 1
            stats[phase]['total_duration'] += duration
            if duration > stats[phase]['max_duration']:
                stats[phase]['max_duration'] = duration
            stats[phase]['total_comps'] += comps
            if comps > stats[phase]['max_comps']:
                stats[phase]['max_comps'] = comps
            
            total_time += duration
            total_comps += comps
            
    outq += ["="]
    outq += [f"SUMMARY FOR: {file_path}"]
    outq += [f"Parallelism: {cards} Cards"]
    outq += ["="]
    outq += [f"{'Phase Name':<20} | {'Configs':<7} | {'Total Dur (s)':<15} | {'Avg Dur (s)':<13} | {'Max Dur (s)':<13} | {'Total Comps':<11} | {'Avg Comps':<13} | {'Max Comps':<11}"]
    outq += ["-"]
    
    total_dur = 0
    total_comps = 0
    max_dur = 0
    max_comps = 0
    for phase in sorted(stats.keys()):
        data = stats[phase]
        count = data['count']
        t_dur = data['total_duration']
        total_dur += t_dur
        t_comp = data['total_comps']
        total_comps += t_comp
        m_dur = data['max_duration']
        if m_dur > max_dur:
            max_dur = m_dur
        m_comp = data['max_comps']
        if m_comp > max_comps:
            max_comps = m_comp
        
        # Averages per configuration/step
        a_dur = t_dur / count if count > 0 else 0
        a_comp = t_comp / count if count > 0 else 0
        
        outq += [f"{phase:<20} | {count:<7} | {(t_dur / cards):<15.4f} | {a_dur:<13.4f} | {m_dur:<13.4f} | {(t_comp / cards):<11} | {a_comp:<13.4f} | {m_comp:<11}"]
        
    outq += ["-"]
    total_configs = sum(d['count'] for d in stats.values())
    avg_total_dur = total_time / total_configs if total_configs > 0 else 0
    avg_total_comps = total_comps / total_configs if total_configs > 0 else 0
    outq += [f"{'TOTAL / WARMUP MEAN':<20} | {total_configs:<7} | {(total_time / cards):<15.4f} | {avg_total_dur:<13.4f} | {max_dur:<13.4f} | {(total_comps / cards):<11} | {avg_total_comps:<13.4f} | {max_comps:<11}"]
    outq += ["="]

    max_out = max(len(line) for line in outq)
    print("")
    for idx, val in enumerate(outq):
        if val in ["=", "-"]:
            print(val * max_out + val + '|')
        else:
            print(val + " " * (max_out - len(val)) + ' |')
    print("")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python summarize_compile_times.py <file1.csv> [num_cards] [file2.csv ...]")
    else:
        cards = int(sys.argv[1])
        for csv_file in sys.argv[2:]:
            summarize_compile_times(csv_file, cards)