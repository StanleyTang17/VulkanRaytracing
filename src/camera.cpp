#include "camera.h"

Camera::Camera() {
	position = glm::vec3(0.0f);
	front = glm::vec3(0.0f, 0.0f, -1.0f);
	worldUp = glm::vec3(0.0f, 1.0f, 0.0f);
	sensitivity = 2.0f;

	pitch = 0.0f;
	yaw = -90.0f;
	roll = 0.0f;

	vfov = 90.0f;

	frontMove = 0;
	sideMove = 0;
	verticalMove = 0;
	speed = 3.0f;

	this->updateVectors();
}

void Camera::updateVectors() {
	front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	front.y = sin(glm::radians(pitch));
	front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

	front = glm::normalize(front);
	right = glm::normalize(glm::cross(front, worldUp));
	up = glm::normalize(glm::cross(right, front));
}

void Camera::handleMouseInput(const float dt, const double offsetX, const double offsetY) {
	pitch += static_cast<float>(offsetY) * sensitivity * dt;
	yaw += static_cast<float>(offsetX) * sensitivity * dt;

	if (pitch > 89.0f)
		pitch = 89.0f;
	else if (pitch < -80.0f)
		pitch = -80.0f;

	if (yaw > 180.0f)
		yaw -= 360.0f;
	else if (yaw < -180.0f)
		yaw += 360.0f;
}

void Camera::handleKeyInput(GLFWwindow* const window, const int key, const int action) {
	if (key == GLFW_KEY_W) {
		if (action == GLFW_PRESS)
			frontMove = 1;
		else if (action == GLFW_RELEASE) {
			if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
				frontMove = -1;
			else
				frontMove = 0;
		}
	} else if (key == GLFW_KEY_S) {
		if (action == GLFW_PRESS)
			frontMove = -1;
		else if (action == GLFW_RELEASE) {
			if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
				frontMove = 1;
			else
				frontMove = 0;
		}
	} else if (key == GLFW_KEY_D) {
		if (action == GLFW_PRESS)
			sideMove = 1;
		else if (action == GLFW_RELEASE) {
			if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
				sideMove = -1;
			else
				sideMove = 0;
		}
	} else if (key == GLFW_KEY_A) {
		if (action == GLFW_PRESS)
			sideMove = -1;
		else if (action == GLFW_RELEASE) {
			if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
				sideMove = 1;
			else
				sideMove = 0;
		}
	} else if (key == GLFW_KEY_SPACE) {
		if (action == GLFW_PRESS)
			verticalMove = 1;
		else if (action == GLFW_RELEASE) {
			if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
				verticalMove = -1;
			else
				verticalMove = 0;
		}
	} else if (key == GLFW_KEY_LEFT_SHIFT) {
		if (action == GLFW_PRESS)
			verticalMove = -1;
		else if (action == GLFW_RELEASE) {
			if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
				verticalMove = 1;
			else
				verticalMove = 0;
		}
	}
}

void Camera::move(const float dt) {
	glm::vec3 velocity(0.0f);

	if (frontMove || sideMove || verticalMove) {
		glm::vec3 frontTranslate = this->front * static_cast<float>(frontMove);
		glm::vec3 sideTranslate = this->right * static_cast<float>(sideMove);
		glm::vec3 verticalTranslate = this->up * static_cast<float>(verticalMove);

		glm::vec3 translate = glm::vec3(
			frontTranslate.x + sideTranslate.x,
			verticalTranslate.y,
			frontTranslate.z + sideTranslate.z
		);
		translate = glm::normalize(translate);

		velocity = translate * speed;
	}

	position += velocity * dt;
}