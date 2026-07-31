from __future__ import annotations

import time


class FreedriveController:
    """Switch the UR motion controller to freedrive and keep it enabled."""

    def __init__(self, node, config):
        from controller_manager_msgs.srv import ListControllers, SwitchController
        from std_msgs.msg import Bool

        self.node = node
        self.config = config
        self._SwitchController = SwitchController
        self._ListControllers = ListControllers
        self._Bool = Bool
        self._client = node.create_client(
            SwitchController, config["controller_manager_switch_service"]
        )
        manager=config["controller_manager_switch_service"].rsplit("/",1)[0]
        self._list_client=node.create_client(ListControllers,f"{manager}/list_controllers")
        self._publisher = node.create_publisher(
            Bool, config["enable_topic"], 10
        )
        self._timer = None
        self.enabled = False

    def controller_states(self):
        import rclpy

        timeout=float(self.config["switch_timeout_sec"])
        if not self._list_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("controller list service unavailable")
        future=self._list_client.call_async(self._ListControllers.Request())
        rclpy.spin_until_future_complete(self.node,future,timeout_sec=timeout)
        if not future.done() or future.result() is None:
            raise RuntimeError("controller state query timed out")
        return {x.name:x.state for x in future.result().controller}

    def _switch(self, activate, deactivate):
        import rclpy

        timeout = float(self.config["switch_timeout_sec"])
        if not self._client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(
                f"controller switch service unavailable: "
                f"{self.config['controller_manager_switch_service']}"
            )
        states=self.controller_states()
        activate=[name for name in activate if states.get(name)!="active"]
        deactivate=[name for name in deactivate if states.get(name)=="active"]
        if not activate and not deactivate:return
        request = self._SwitchController.Request()
        request.activate_controllers = list(activate)
        request.deactivate_controllers = list(deactivate)
        request.strictness = self._SwitchController.Request.STRICT
        request.activate_asap = True
        request.timeout.sec = int(timeout)
        request.timeout.nanosec = int((timeout - int(timeout)) * 1_000_000_000)
        future = self._client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout + 0.5)
        if not future.done():
            raise RuntimeError("controller switch timed out")
        result = future.result()
        if result is None or not result.ok:
            detail = getattr(result, "message", "") if result is not None else ""
            raise RuntimeError(f"controller switch failed: {detail or 'no result'}")
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            states=self.controller_states()
            if all(states.get(x)=="active" for x in activate) and all(states.get(x)!="active" for x in deactivate):
                return
            time.sleep(0.05)
        detail=", ".join(f"{name}={states.get(name,'missing')}" for name in [*activate,*deactivate])
        raise RuntimeError(f"controller switch response was OK but lifecycle state did not converge: {detail}")

    def _publish(self, value):
        msg = self._Bool()
        msg.data = bool(value)
        self._publisher.publish(msg)

    def enable(self):
        self._switch(
            [self.config["controller_name"]],
            [self.config["motion_controller_name"]],
        )
        self.enabled = True
        self._publish(True)
        self._timer = self.node.create_timer(
            1.0 / float(self.config["keepalive_rate_hz"]),
            lambda: self._publish(True),
        )

    def disable_and_restore(self):
        if not self.enabled:
            return
        if self._timer is not None:
            self._timer.cancel()
            self.node.destroy_timer(self._timer)
            self._timer = None
        self._publish(False)
        # Give the controller manager one update opportunity to receive False.
        time.sleep(0.05)
        try:
            self._switch(
                [self.config["motion_controller_name"]],
                [self.config["controller_name"]],
            )
        finally:
            self.enabled = False

    def ensure_motion_controller(self):
        """Make the configured trajectory controller active without commanding motion."""
        self._publish(False)
        self._switch(
            [self.config["motion_controller_name"]],
            [self.config["controller_name"]],
        )
