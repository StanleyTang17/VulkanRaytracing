#include "font.h"

Font::Font() {
	std::fill(pixels, pixels + TEXTURE_SIZE, 0);
}

Font::Font(const char* fontFamilyPath, const unsigned int fontSize)
	:
	fontSize(fontSize)
{
	loadFont(fontFamilyPath, fontSize);
}

void Font::loadFont(const char* fontFamilyPath, const unsigned int fontSize)
{
	this->fontSize = fontSize;

	FT_Library ft;
	if (FT_Init_FreeType(&ft)) {
		throw std::runtime_error("failed to load font library!");
	}

	FT_Face face;
	if (FT_New_Face(ft, fontFamilyPath, 0, &face)) {
		throw std::runtime_error("failed to load font face!");
	}

	FT_Set_Pixel_Sizes(face, 0, fontSize);

	std::fill(pixels, pixels + TEXTURE_SIZE, 0);

	for (unsigned char c = 0; c < 128; ++c) {
		Character character;
		if (loadChar(character, face, c)) {
			charData.emplace(c, character);
		}
	}

	FT_Done_Face(face);
	FT_Done_FreeType(ft);

	loaded = true;
}

bool Font::loadChar(Character& character, FT_Face& face, char ch) {
	if (FT_Load_Char(face, ch, FT_LOAD_RENDER)) {
		throw std::runtime_error("failed to load font character");
	}

	character.size = glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows);
	character.bearing = glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top);
	character.advance = face->glyph->advance.x;

	size_t glyphTopLeftX = static_cast<size_t>(ch) % NUM_CHARS_PER_ROW * GLYPH_WIDTH;
	size_t glyphTopLeftY = static_cast<size_t>(ch) / NUM_CHARS_PER_ROW * GLYPH_HEIGHT;

	// Top left
	character.texCoords[0] = glm::vec2(
		static_cast<float>(glyphTopLeftX) / TEXTURE_WIDTH,
		static_cast<float>(glyphTopLeftY + character.size.y) / TEXTURE_HEIGHT
	);
	// Top right
	character.texCoords[1] = glm::vec2(
		static_cast<float>(glyphTopLeftX + character.size.x) / TEXTURE_WIDTH,
		static_cast<float>(glyphTopLeftY + character.size.y) / TEXTURE_HEIGHT
	);
	// Bottom right
	character.texCoords[2] = glm::vec2(
		static_cast<float>(glyphTopLeftX + character.size.x) / TEXTURE_WIDTH,
		static_cast<float>(glyphTopLeftY) / TEXTURE_HEIGHT
	);
	// Bottom left
	character.texCoords[3] = glm::vec2(
		static_cast<float>(glyphTopLeftX) / TEXTURE_WIDTH,
		static_cast<float>(glyphTopLeftY) / TEXTURE_HEIGHT
	);

	for (size_t bitmapX = 0; bitmapX < character.size.x; ++bitmapX) {
		for (size_t bitmapY = 0; bitmapY < character.size.y; ++bitmapY) {
			size_t pixel_x = glyphTopLeftX + bitmapX;
			size_t pixel_y = glyphTopLeftY + bitmapY;
			size_t pixel_index = pixel_y * NUM_CHARS_PER_ROW * GLYPH_WIDTH + pixel_x;
			size_t bitmap_index = bitmapY * character.size.x + bitmapX;

			pixels[pixel_index] = face->glyph->bitmap.buffer[bitmap_index];
		}
	}

	return true;
}