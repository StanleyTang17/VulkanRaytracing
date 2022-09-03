#ifndef CAMERA_H
#define CAMERA_H

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include <glfw/glfw3.h>

class Camera
{
private:
	glm::vec3 worldUp;
	glm::vec3 position;
	glm::vec3 front;
	glm::vec3 right;
	glm::vec3 up;

	float sensitivity;
	glm::vec3 velocity;

	float pitch;
	float yaw;
	float roll;

	float vfov;

	int frontMove;
	int sideMove;
	int verticalMove;
	float speed;

public:
	Camera();

	void updateVectors();
	void handleMouseInput(const float dt, const double offsetX, const double offsetY);
	void handleKeyInput(GLFWwindow* const window, const int key, const int action);
	void move(const float dt);

	inline void setPosition(glm::vec3 position) { this->position = position; }
	inline void setWorldUp(glm::vec3 worldUp) { this->worldUp = worldUp; }
	inline void setFront(glm::vec3 front) { this->front = front; }
	inline void setSensitivity(float sensitivity) { this->sensitivity = sensitivity; }
	inline void setVFOV(float vfov) { this->vfov = vfov; }
	inline void setSpeed(float speed) { this->speed = speed; }

	inline glm::vec3 getPosition() const { return position; }
	inline glm::vec3 getFront() const { return front; }
	inline glm::vec3 getRight() const { return right; }
	inline glm::vec3 getUp() const { return up; }
	inline float getVFOV() const { return vfov; }
};

#endif