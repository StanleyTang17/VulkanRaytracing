#ifndef FONT_H
#define FONT_H

#include <ft2build.h>
#include FT_FREETYPE_H
#include <glm/glm.hpp>
#include <unordered_map>
#include <stdexcept>

struct Character
{
	glm::ivec2 size;
	glm::ivec2 bearing;
	unsigned int advance;
	glm::vec2 texCoords[4]; // { topLeft, topRight, bottomRight, bottomLeft }
};

class Font
{
public:
	static const unsigned int GLYPH_WIDTH = 25;
	static const unsigned int GLYPH_HEIGHT = 25;
	static const unsigned int NUM_CHARS_PER_ROW = 16;
	static const unsigned int NUM_ROWS = 8;
	static const unsigned int TEXTURE_WIDTH = NUM_CHARS_PER_ROW * GLYPH_WIDTH;
	static const unsigned int TEXTURE_HEIGHT = NUM_ROWS * GLYPH_HEIGHT;
	static const size_t TEXTURE_SIZE = TEXTURE_WIDTH * TEXTURE_HEIGHT;

	Font();
	Font(const char* fontFamilyPath, const unsigned int fontSize);

	void loadFont(const char* fontFamilyPath, const unsigned int fontSize);

	inline unsigned char* getFontPixels() { return pixels; }
	inline Character getCharacter(const char ch) const { return charData.at(ch); }
	inline unsigned int getFontSize() { return fontSize; }

private:
	bool loaded = false;
	unsigned char pixels[TEXTURE_SIZE];
	std::unordered_map<char, Character> charData;
	unsigned int fontSize;
	
	bool loadChar(Character& character, FT_Face& face, char ch);
};

#endif