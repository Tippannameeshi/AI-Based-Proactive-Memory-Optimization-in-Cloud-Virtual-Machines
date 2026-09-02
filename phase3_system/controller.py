import requests
import time

VMS = {
    "testvm1": "192.168.122.213",
    "testvm2": "192.168.122.109",
    "testvm3": "192.168.122.104"
}


def get_vm_status(name, ip):
    try:
        cpu = requests.get(
            f"http://{ip}:5000/cpu",
            timeout=3
        ).json()

        memory = requests.get(
            f"http://{ip}:5000/memory",
            timeout=3
        ).json()

        cpu_percent = cpu["cpu_percent"]
        memory_percent = memory["usage_percent"]

        # Load score: lower score = better VM
        score = (cpu_percent * 0.5) + (memory_percent * 0.5)

        return {
            "name": name,
            "ip": ip,
            "status": "ONLINE",
            "cpu": cpu_percent,
            "memory": memory_percent,
            "score": round(score, 2)
        }

    except Exception:
        return {
            "name": name,
            "ip": ip,
            "status": "OFFLINE",
            "cpu": 0,
            "memory": 0,
            "score": 9999
        }


def monitor_vms():
    results = []

    for name, ip in VMS.items():
        results.append(get_vm_status(name, ip))

    return results


def select_best_vm(results):
    online_vms = [
        vm for vm in results
        if vm["status"] == "ONLINE"
    ]

    if not online_vms:
        return None

    return min(online_vms, key=lambda vm: vm["score"])


def send_task(vm, task):
    try:
        response = requests.post(
            f"http://{vm['ip']}:5000/task",
            json={"task": task},
            timeout=15
        )

        return response.json()

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


def display_monitor(results):
    print("\n" + "=" * 75)
    print("           DISTRIBUTED VM MONITOR & INTELLIGENT SCHEDULER")
    print("=" * 75)

    print(
        f"{'VM':<12}"
        f"{'STATUS':<12}"
        f"{'CPU':<12}"
        f"{'MEMORY':<12}"
        f"{'SCORE':<12}"
    )

    print("-" * 75)

    for vm in results:
        print(
            f"{vm['name']:<12}"
            f"{vm['status']:<12}"
            f"{str(vm['cpu']) + '%':<12}"
            f"{str(vm['memory']) + '%':<12}"
            f"{vm['score']:<12}"
        )

    print("=" * 75)


def main():
    print("\nStarting Distributed VM Scheduler...\n")

    results = monitor_vms()

    display_monitor(results)

    best_vm = select_best_vm(results)

    if best_vm:
        print("\n>>> SCHEDULER DECISION")
        print(f">>> SELECTED VM: {best_vm['name']}")
        print(f">>> IP ADDRESS: {best_vm['ip']}")
        print(f">>> LOAD SCORE: {best_vm['score']}")

        print("\n>>> SUBMITTING TASK...")

        # Demo task
        task = "echo 'Task executed successfully on $(hostname)'"

        result = send_task(best_vm, task)

        print("\n>>> TASK RESULT")
        print(f"STATUS: {result.get('status')}")
        print(f"HOSTNAME: {result.get('hostname')}")
        print(f"OUTPUT: {result.get('stdout')}")
        print(f"EXECUTION TIME: {result.get('execution_time')} seconds")
        print(f"RETURN CODE: {result.get('return_code')}")

    else:
        print("\nNo online VMs available!")


if __name__ == "__main__":
    main()
