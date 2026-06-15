cmake_minimum_required(VERSION 3.8)
project(vehicle_nmpc_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(srvs
  "srv/ResetSimulation.srv"
  "srv/StepSimulation.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${srvs}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
